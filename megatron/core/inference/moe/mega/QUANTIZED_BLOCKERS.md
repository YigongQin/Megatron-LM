# What blocks mxfp8 / nvfp4 / fp8_fp4 on the mega MoE path

`inference_grouped_gemm_backend='flashinfer_mega'` accepts four precisions via
`inference_mega_precision`: `bf16`, `mxfp8`, `nvfp4`, `fp8_fp4`. All four run,
and all four have been benchmarked and correctness-checked as standalone
generation.

**`bf16` and `mxfp8` are usable for RL train/generation parity. `nvfp4` and
`fp8_fp4` are not.** mxfp8 was blocked until §1 was resolved for it; what
follows records both what blocked it and why the way out turned out to be
cheaper than this document originally estimated.

This is the list of what stands in the way, separated by which of the two
use cases it blocks, because they are blocked for different reasons and one of
them is a design decision rather than missing work.

## 1. Refit cannot rebuild a quantized kernel weight — blocks RL generation

The hard one. The megakernel does not read Megatron's parameters at forward
time; it reads its own transformed copy of the expert weights. For bf16 that
copy is ours (`MegaKernelWeightBuffer`), which is what makes a refit a repack:
`InferenceGroupedMLP.refresh_mega_weights` marks the buffer stale and the next
forward rewrites it in place, with no teardown, no EP collective and no CuTeDSL
recompile.

The quantized precisions cannot use that buffer. FlashInfer's
`preprocess_mega_weights` does two things — it interleaves gate/up in blocks of
32 and transposes to K-major, *and* it quantizes — and `MegaKernelWeightBuffer`
reproduces only the first. There is no layout we can write bf16 parameters into
that a quantized kernel will read. So those precisions keep
`preprocess_weights=True`, FlashInfer snapshots the weights at construction and
releases the source, and a subsequent refit never reaches the kernel.

`refresh_mega_weights` raises `NotImplementedError` for them rather than
returning quietly. That matters: the failure mode it replaces is silent. Nothing
reads the expert parameters again, so generation would keep sampling from the
weights snapshotted before the refit — finite, plausible rollouts from the
previous step's policy, and the importance ratio that exists to catch
train/generation mismatch would be computed against those same stale log-probs
and would look clean.

Ways out, in increasing order of work:

- **Rebuild the adapter per refit.** Drop `MegatronMegaMoEAdapter._layer` and
  let the next forward reconstruct it. Correct and small, but it pays an EP
  collective, a symmetric-heap reallocation and a CuTeDSL recompile on every
  refit, and construction is illegal under CUDA graph capture, so every refit
  would also invalidate the graphs.
- **Quantize into a caller-owned buffer.** *This is what mxfp8 now does.* The
  estimate here was that it means "reproducing its scale derivation bit-for-bit
  per precision and keeping it in step with FlashInfer" — four reimplementations
  to maintain. That was wrong, and the correction is the reusable part: the
  quantizer and the scale swizzle are exported from the `cutedsl_megamoe` shim
  (`mxfp8_quantize_per_block_32`, `to_blocked`, `_stack_byte_reinterpretable_tensors`,
  all in its `__all__`), which is the same boundary FlashInfer's own backends
  import them through. So `MegaMxfp8KernelWeightBuffer` *calls* them rather than
  reimplementing them, and agreement is by construction rather than by
  maintenance.

  Two other things made mxfp8 cheap. The quantizer's input orientation is
  exactly what the bf16 buffer already produces — FlashInfer interleaves
  gate/up, transposes to K-major, then quantizes per expert — so
  `MegaKernelWeightBuffer.views()` is the right input unchanged. And MXFP8
  weight quantization uses a fixed 1.0 norm with no global amax, so there is no
  calibration state to keep in sync. `test_mega_training_weights.py`,
  `TestMxfp8KernelWeights` pins the result byte-for-byte against
  `preprocess_mega_weights` and checks FlashInfer's own
  `validate_transformed_mega_weights` accepts it.

  nvfp4 is not blocked by anything new after this, but it has not been done: it
  needs its own pinning test, and its second-level scale (`input_norm_const`,
  see §2a) is a trap that mxfp8 does not have.
- **Ask FlashInfer for an in-place reload.** A `MoEEpMegaLayer` method that
  re-runs preprocessing into the existing transformed tensors, keeping the
  workspace and the compiled kernel. This is the right place for it — the
  quantizer lives there and stays in step by construction — and it is one
  upstream request rather than four reimplementations.

## 2. A quantized forward against a bf16 backward — blocks the training forward

`moe_mega_training_forward` is rejected for anything but bf16 in
`TransformerConfig.__post_init__` **unless
`moe_mega_training_straight_through=True`**. This is a deliberate refusal that
can be waived, not a gap — and the reasoning below is why it is a waiver with a
name rather than an implicit consequence of choosing a precision.

The parity hybrid runs the megakernel for the forward value and takes the
gradient from the MoE recompute pass, which runs the ordinary TE bf16
dispatch + grouped-GEMM path. In bf16 the two differ only by reduction order
and rounding. With a quantized forward they differ by the quantization itself:
the gradient would be of the bf16 function, not of the function that produced
the value. That is a straight-through estimator, which is a training-recipe
decision with convergence consequences — not something a backend flag should
turn on implicitly.

If quantized generation is wanted alongside a bf16 training forward, the honest
framing is that train/generation parity is being given up on purpose, and the
resulting bias belongs in the recipe. Nothing here prevents that combination
from being built; it just should not arrive via `moe_mega_training_forward`.

**What `moe_mega_training_straight_through` buys, and what it does not.** With
mxfp8 on *both* sides, train/generation parity is not given up at all: the same
kernel runs at the same precision on the same weights, so the two forwards are
bitwise identical — measured at cold=0, warm=0 across repeated launches by
`TestTrainGenParity::test_training_forward_matches_generation_forward[mxfp8]`.
The importance ratio stays clean. What is given up is gradient fidelity, exactly
as described above: the gradient is of the bf16 function while the value is
mxfp8. Convergence against the bf16 arm has not been measured. So the flag is
not a correctness escape hatch — it is the recipe stating that it accepts a
straight-through estimator in exchange for a quantized forward that still has
parity.

## 2a. Batch invariance and determinism are *not* what blocks them

Measured, so it does not get re-litigated. At 128 experts, EP=4, 1024 tokens and
three router skews, both quantized kernels are bitwise stable:

| precision | wide vs decode-width | 8 identical relaunches | two built instances |
| --- | --- | --- | --- |
| `bf16` | 0 tokens differ | 0 differ | 0 tokens differ |
| `mxfp8` | 0 tokens differ | 0 differ | 0 tokens differ |
| `nvfp4` | 0 tokens differ | 0 differ | 0 tokens differ |

`./scripts/local/run_mega_training_tests.sh --precision all` reproduces it
(`TestKernelBatchInvarianceByPrecision`, `TestKernelDeterminismByPrecision`).
The tests assert zero differences rather than a tolerance, so the table is what
a pass means; re-run it rather than trusting these cells, since the phase log is
overwritten per run and only the nvfp4 one was kept.

nvfp4 passing is worth explaining, because the usual NVFP4 recipe would fail it.
A calibrated or dynamic per-tensor amax for the second-level scale makes a
token's quantization depend on what else is in the launch. FlashInfer does not
compute one: the second-level scale is `input_norm_const`, a static float on the
kernel config (default 1.0), handed to `stage_mega_moe_inputs`, and the first
level is `nvfp4_quantize_per_block_16`, whose fp8 scale depends only on the 16
values inside its own block. Neither level reads outside the token. Weight
scales are frozen at preprocessing with `norm_const=1.0`.

**So `input_norm_const` must stay static.** It is a plain config field and
setting it from a measured amax — an obvious-looking accuracy fix — silently
ends batch invariance. The symptom would be a small fraction of tokens
disagreeing between the rollout and the log-prob pass, which is the same
signature as the expert-ordering bug and took days to attribute. `registry.py`
does not set it.

Two things this does *not* establish. It says the kernels are reproducible, not
that they are accurate: with `input_norm_const` left at 1.0 and uncalibrated,
nvfp4 accuracy rests entirely on the per-block-16 scales, and the forward error
against TE bf16 is a separate measurement. And it does not touch §1 — both
precisions still snapshot their weights at construction, so refit remains the
blocker for RL.

## 3. Megatron-side MXFP8 is a separate thing, and the two do not compose

Easy to conflate. Two different MXFP8s are in play:

- **Megatron/TE MXFP8** (`--fp8` with `fp8_recipe=mxfp8`), where expert weights
  are stored as `MXFP8Tensor`. Config validation rejects this with
  `flashinfer_mega`: the mega weight packer cannot read an `MXFP8Tensor`.
- **Kernel-side MXFP8** (`inference_mega_precision='mxfp8'`), where parameters
  stay bf16 and FlashInfer quantizes them during preprocessing. This is the one
  the mega path uses.

So a mega MXFP8 run keeps `--fp8` off entirely. Related, and worth knowing
before reaching for TE MXFP8 as an alternative: `InferenceGroupedMLP.forward`
asserts MXFP8 inference-optimized is incompatible with training, so that route
is closed for colocated RL regardless of the megakernel.

## 4. Practicalities, none of them blocking

- **`fp8_fp4` needs DeepGEMM.** It is the only mega kernel not written in
  CuTeDSL; FlashInfer ships the wrapper but does not depend on the package.
  `registry.py` checks `importlib.util.find_spec("deep_gemm")` at config-build
  time so the failure names the package instead of surfacing from inside the
  first forward. Installing it from source needs `libdw-dev` for
  `elfutils/libdwfl.h`, and its `install.sh` calls a bare `python`, which does
  not exist in the container — install with `pip --no-build-isolation` instead.
  It is not installed by default anywhere on this stack: mcore keeps it behind
  its own `batch_invariant` extra, and RL lists it only under `vllm`, so the
  `mcore` worker venv does not have it. Deliberate — nothing but `fp8_fp4` wants
  it, and §1 already rules `fp8_fp4` out for RL, so making every worker venv
  build it from source would buy nothing.
- **Shape alignment is per precision**, enforced in `TransformerConfig` as
  `(hidden_size divisor, moe_ffn_hidden_size divisor)`: bf16 `(32, 64)`,
  mxfp8 `(64, 32)`, nvfp4 `(64, 16)`, fp8_fp4 `(128, 32)`. A geometry that fits
  bf16 need not fit nvfp4.
- **SwiGLU only**, under every precision: the kernel stacks gate+up into w13
  and applies SwiGLU itself, so a non-gated activation has no representable
  weight layout. `activation_func_tanh_clamp_scale` is also rejected — the
  kernel offers a hard FC1 clamp, while that flag is a soft tanh clamp
  replacing the swish gate (SiTU-GLU), and the two are not interchangeable.

## Where the tests are

- `tests/unit_tests/inference/test_mega_training_weights.py` pins our bf16
  repack against FlashInfer's `_interleave_gate_up_32` and our mxfp8
  quantization against its `preprocess_mega_weights`
  (`TestMxfp8KernelWeights`), and covers the config validation above. CPU-only
  and single rank: `PHASES=weights`.
- `tests/unit_tests/inference/test_mega_training_forward.py`,
  `TestGenerationWeightOwnership` covers the refit path end to end for both
  owned precisions, including the refusal in §1 for the unowned ones.
  `scripts/local/run_mega_training_tests.sh` with `PHASES=gen` runs just that
  class. `--precision bf16` or `--precision mxfp8` narrows it to one; the flag
  selects precisions and does not change which phases run.
- `TestKernelBatchInvarianceByPrecision` and `TestKernelDeterminismByPrecision`
  in the same file are parametrized over `inference_mega_precision` and produce
  §2a. Run them with `--precision bf16,mxfp8,nvfp4` (or `--precision all`, which
  is the same three — `fp8_fp4` is excluded because no worker venv has DeepGEMM
  and it would only ever skip).
