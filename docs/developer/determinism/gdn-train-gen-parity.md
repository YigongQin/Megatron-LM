---
orphan: true
---

# Train/generation parity for Gated DeltaNet

Analysis and plan for extending `batch_invariant_mode` to the linear-attention
mixers, so a GDN model can reach a zero train/generation log-prob gap the way a
Mamba hybrid now does. Nothing here is implemented; this is a scoping document.

Reinforcement learning compares a training log-prob against a generation
log-prob. Any kernel the two paths do not share is a bias in the gradient, and a
per-token bias does not average away across a rollout. Batch-invariant mode
exists to remove those differences; today it rejects every GDN variant.

## Current support

| variant | training | dynamic inference | `batch_invariant_mode` |
| --- | --- | --- | --- |
| `GatedDeltaNet` (`ssm/gated_delta_net/gdn.py`) | yes | prefill, decode, CUDA graphs | asserted off (`gdn.py:303`) |
| `GatedDeltaNet2` (`gdn2.py`) | yes | `NotImplementedError` (`gdn2.py:182`) | n/a |
| `GatedDeltaProductMixer` (`gated_delta_product.py`) | yes | prefill, decode, speculative | asserted off (`:566`) |

`GatedDeltaNet` is the practical target: it is the only variant with both a
training path and a complete dynamic-inference path. It reuses
`SSMDynamicInferenceMixin`, so the decode/prefill partitioning, slot allocation,
chunk metadata and packed-token merge are shared with Mamba and do not need to
be rebuilt.

## Which kernels each stage runs

| stage | conv | scan | decay gate `g`, `beta`, q/k L2 norm |
| --- | --- | --- | --- |
| training | FLA `causal_conv1d` | FLA `chunk_gated_delta_rule` | outside the kernel, `@jit_fuser` in FP32 |
| training, `deterministic_mode` | `F.conv1d` | `torch_chunk_gated_delta_rule` | outside the kernel |
| dynamic prefill | FLA `causal_conv1d` | FLA `chunk_gated_delta_rule` | inside the kernel |
| dynamic decode | FLA `causal_conv1d_update` | FLA `fused_recurrent_gated_delta_rule` | inside the kernel |

The starting position is better than Mamba's was. Training and prefill already
call the *same* function. Mamba needed `mamba_chunk_scan_combined` (pip) to
agree with `mamba_chunk_scan_combined_varlen` (in-tree), which is what
`tests/unit_tests/ssm/test_mamba_train_gen_parity.py::TestScanParity` pins.

## Gaps

**1. The gate, `beta` and the q/k L2 norm are computed on opposite sides of the
kernel.** Training calls the scan with `use_qk_l2norm_in_kernel=False`
(`gdn.py:252`), deriving `g` and `beta.sigmoid()` in `_compute_gates` under
`@jit_fuser` and normalizing in `_prepare_input_for_gated_delta_rule`. Prefill
and decode both pass `A_log` and `dt_bias` into the kernel with
`use_gate_in_kernel=True`, `use_beta_sigmoid_in_kernel=True` and
`use_qk_l2norm_in_kernel=True` (`gdn.py:398-400`, `:445-447`). Training is the
odd one out. This is the same shape of bug as the Mamba gate placement and the
batch-invariant squared ReLU: identical algebra, different rounding.

**2. `deterministic_mode` makes training and dynamic inference mutually
exclusive.** It swaps training onto `torch_chunk_gated_delta_rule` and `F.conv1d`,
while dynamic inference asserts `not deterministic_mode` (`gdn.py:300`) because it
requires the FLA recurrent kernels. Parity wants the opposite of what the flag
does: both sides on FLA. Note that parity requires a bitwise *forward*, not a
deterministic backward, so these are separable requirements that the current
flag conflates.

**3. Decode runs a recurrence, and there is no buffered chunk replay for GDN.**
This is the substantive work. Mamba only reaches KL=0 because
`ssm/ops/mamba2/batch_invariant_decode.py` (369 lines) buffers decode tokens per
slot and replays the chunk scan at chunk boundaries rather than stepping a
recurrence, calling the in-tree primitives `_chunk_cumsum_fwd`,
`_chunk_state_fwd`, `_state_passing_fwd`, `_chunk_scan_fwd` and `_bmm_chunk_fwd`
directly. GDN's
chunk kernel is FLA, a third-party package, and cannot be decomposed that way.

**4. Packed sequences have no deterministic path**, already recorded in
[`op-catalog.md`](./op-catalog.md). The Mamba work hit the same wall from the
other direction: the unfused training path parity requires rejects packing.

## Plan

**Phase 1 — prefill parity, and a decision point.** Move training onto the
in-kernel gate, sigmoid and L2 norm so it calls the scan exactly as prefill
does; allow `batch_invariant_mode` for GDN when decode is not involved; and
tighten `test_gated_delta_net_inference.py::test_prefill_decode_matches_full_forward`,
which today compares the paths at `atol=rtol=3e-2`, into a bitwise
training-versus-prefill comparison.

This is deliberately the cheap half, and it answers the expensive question:
whether FLA's chunk kernel returns the same bits under `cu_seqlens` varlen
partitioning as under a batch layout. If it does not, phase 2 needs an in-tree
fork regardless, and the estimate roughly doubles. Better to learn that in a day
than after building a replay against a kernel that cannot support one.

**Phase 2 — decode parity.** Port the buffered-replay design onto GDN. The
likely prerequisite is forking FLA's gated-delta-rule chunk kernel in-tree,
which is precisely what already exists for GDP under `ssm/ops/gdp/`
(`chunk_h.py`, `chunk_o.py`, `wy_fast.py`, `solve_tril.py`). The conv side is
comparatively cheap: Mamba hand-matches the training conv arithmetic in its
decode step rather than calling `causal_conv1d_update`.

Phase 2 is required, not optional. Generation log-probs come from decode, which
is why Mamba needed `MambaBatchInvariantDecode` at all.

## Risks

- FLA is third-party, so every kernel-level fix is either an upstream change, a
  fork, or a constraint on the pinned version.
- GDP is the opposite trade: its prefill and decode kernels are already in-tree,
  but training runs FLA or cuTeDSL, so training and generation share nothing. It
  may be the better vehicle for the replay and the worse one for phase 1.
- `test_gated_delta_net_inference.py` runs at `atol=rtol=3e-2` today. Tightening
  it to bitwise may surface differences unrelated to this work.

## Note

[`op-catalog.md`](./op-catalog.md) rows 39 and 70 point at
`megatron/core/ssm/gated_delta_net.py`. That module is now the package
`megatron/core/ssm/gated_delta_net/`; the kernel selection is in `gdn.py` and
the packed-sequence assertion is at `gdn.py:129`.
