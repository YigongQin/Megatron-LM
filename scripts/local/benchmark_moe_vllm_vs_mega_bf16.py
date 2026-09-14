#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Benchmark + correctness-check inference-optimized MoE: vLLM/torch vs mega BF16.

Requires multi-GPU EP (torchrun) and Blackwell for the mega path. The sm100 BF16
CuTeDSL megakernel is unreleased as of flashinfer 0.6.x (targeted for 0.7.0), so
the mega path needs flashinfer built from a source tree that contains
flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_bf16_bf16_cutedsl.

All backends run the same SwiGLU geometry, which is all the megakernel implements.

What is measured
----------------
One ``MoELayer`` from ``get_inference_optimized_moe_spec()`` — the whole routed
MoE layer, not just the grouped GEMM. The timed region covers router, dispatch,
expert compute and combine. Router and pre/post-processing are identical across
backends, so the isolated expert+EP speedup is larger than the reported
end-to-end ratio.

    hidden_states [T_local, 1, H] bf16
        |
        v
    MoELayer.route ............ InferenceTopKRouter
        |                       -> probs       [T_local, topk] fp32
        |                       -> routing_map [T_local, topk] expert ids
        v
    MoELayer.preprocess ....... [T_local, 1, H] -> [T_local, H]
        |
        v
    MoELayer.dispatch  <<<<<<<< backends diverge
        |
        |-- vllm / torch: NCCLAllGatherDispatcher
        |     AllGather hidden + probs + routing_map over EP -> [T_global, H]
        |
        `-- flashinfer_mega: MegaLocalPassthroughDispatcher
              no-op; tokens stay local, only the valid-token count is recorded
        |
        v
    MoELayer.routed_experts_compute ..... InferenceGroupedMLP.forward
        |
        |-- vllm:  _vllm_forward ............ Triton grouped GEMM (from vLLM)
        |-- torch: _mcore_fused_moe_forward . torch.nn.functional.grouped_mm
        |     both: local experts only, fc1 -> SwiGLU -> fc2, valid_tokens gated
        |
        `-- flashinfer_mega: _mega_forward -> flashinfer MoEEpMegaLayer
              EP all-to-all + fc1 + SwiGLU + fc2 + topk combine, one fused kernel
        |
        v
    MoELayer.combine  <<<<<<<<< backends diverge
        |
        |-- vllm / torch: ReduceScatter -> [T_local, H]
        `-- flashinfer_mega: no-op (the megakernel already reduced across EP)
        |
        v
    output [T_local, 1, H] bf16

The diagram is the whole layer: no shared expert and no expert TP.

Grouped GEMM shapes
-------------------
Symbols: H = --hidden-size, I = --moe-ffn-hidden-size (post-SwiGLU width),
T = --local-tokens per rank, E = --num-experts, k = --topk,
EP = --expert-parallel-size, E_loc = E / EP experts owned per rank.

Routing sends T*EP*k token-expert pairs to E experts, so a rank receives
m_rank ~= T*k rows spread over its E_loc experts (m_e ~= T*k/E_loc per expert,
on average -- the real split is data-dependent and is exactly what makes these
GEMMs "grouped": one GEMM per expert, all with different m_e).

    per-expert e, m_e rows:
      FC1 (gate+up)  [m_e, H]  @ [H, 2I]  -> [m_e, 2I]
      SwiGLU         [m_e, 2I]            -> [m_e, I]
      FC2 (down)     [m_e, I]  @ [I, H]   -> [m_e, H]

    weight stacks held per rank (Megatron layout, out-features major):
      _fc1_weight [E_loc, 2I, H]      w13 in FlashInfer's MoEWeightPack
      _fc2_weight [E_loc,  H, I]      w2

    sum(m_e) over local experts = m_rank ~= T*k

Worked examples, per expert:

  default toy shape (H=128, I=128, E=8, k=2, T=32, EP=4 -> E_loc=2)
      m_rank ~= 64, m_e ~= 32
      FC1 [32, 128] @ [128, 256] -> [32, 256]
      FC2 [32, 128] @ [128, 128] -> [32, 128]
      weights/rank: 2 experts * 0.1 MiB = 0.2 MiB
    Latency-bound; measures launch and transport overhead, not GEMM.

  --preset dsv3 (H=7168, I=2048, E=256, k=8, EP=4 -> E_loc=64), T=128
      m_rank ~= 1024 rows over 64 experts, m_e ~= 16
      FC1 [16, 7168] @ [7168, 4096] -> [16, 4096]
      FC2 [16, 2048] @ [2048, 7168] -> [16, 7168]
      weights/rank: 64 experts * 84 MiB = 5.25 GiB (bf16)
    Note m_e ~= 16 against K=7168: these are extremely skinny GEMMs, which is
    the realistic decode regime and exactly where the mega kernel's fused
    transport is supposed to pay off. Raise --local-tokens to move toward a
    prefill-shaped, compute-bound regime.

Note the m dimension differs by backend even at identical settings: vllm/torch
AllGather first, so each rank's experts see rows drawn from all T*EP global
tokens, while the mega kernel keeps tokens local and moves them inside the
kernel. The totals match; only where the gather happens differs.

Fixed configuration (_make_config)
----------------------------------
    num_layers                          1     single MoE layer, no attention
    moe_router_score_function      softmax
    moe_router_dtype                 fp32     required by inference_optimized
    activation                     SwiGLU     F.silu + gated_linear_unit=True
    normalization                  RMSNorm    add_bias_linear=False
    dtype                             bf16    params_dtype=torch.bfloat16
    transformer_impl   inference_optimized
    expert_tensor_parallel_size          1
    moe_shared_expert_intermediate_size  None  shared expert disabled

CLI-tunable
-----------
    --preset                dsv3 -> H=7168, I=2048, E=256, k=8 (+ --gpu-init)
    --hidden-size           H, default 128
    --moe-ffn-hidden-size   post-SwiGLU expert width, default 128
                            (alignment is per precision: bf16 %64, mxfp8 %32,
                             nvfp4 %16, fp8_fp4 %32; hidden has its own rule)
    --num-experts           E, default 8 (must be divisible by EP)
    --topk                  k, default 2
    --expert-parallel-size  EP, default = world size
    --local-tokens          tokens per rank, default 32
    --mega-max-tokens       mega per-rank cap, default 128
    --gpu-init              init experts on GPU; CPU init costs minutes at
                            model scale
    --warmup / --iters      timing loop, default 5 / 30
    --vllm-dispatcher       nccl | nvls (baseline only; mega is always passthrough)
    --backend               both | vllm | flashinfer_mega
    --mega-precision        bf16 | mxfp8 | nvfp4 | fp8_fp4 (mega kernel only)
    --backend               auto | all | vllm | torch | flashinfer_mega
    --check / --check-tol   correctness mode (reference is always torch:bf16)

Correctness mode (--check)
--------------------------
Every backend runs on the reference backend's weights and the same input, then is
compared against the reference. Weights are copied before the first forward,
because the concatenated-weight cache and the mega weight snapshot are both built
lazily on first call. Results are reported per rank, since a wrong
expert-to-rank mapping corrupts only the ranks owning the misindexed experts.
The process exits nonzero on failure.

The reference is always torch:bf16 -- the eager in-tree grouped GEMM with no
quantization. Everything is measured against it, which is what makes the
variants comparable to each other. Every variant is seeded from one BF16
weight snapshot and applies its own quantization.

What runs alongside the mega kernel, by default:

  vllm:bf16     Megatron's BF16 Triton fused MoE.
  torch:mxfp8   Megatron's own MXFP8 grouped GEMM (fp8_recipe=mxfp8 +
                fp8_param, converted by quantize_model_to_mxfp8). Included
                automatically when --mega-precision is a format Megatron also
                implements, which today means mxfp8 only.

                Caveat for the timing number, not the error number: Megatron's
                fused permute/activation+quantize kernels are squared-ReLU
                only, so under the SwiGLU the mega kernel requires, this runs
                with inference_moe_disable_fused_quant_kernels=True. Both
                GEMMs are still MXFP8 and the arithmetic is unchanged, but the
                quantize is a separate launch, so torch:mxfp8 here is slower
                than Megatron's MXFP8 at its best activation. Read the speed
                ratio as an upper bound on mega's advantage.

So at --mega-precision mxfp8 you get mega:mxfp8 and torch:mxfp8 measured
against the same BF16 oracle, and the run prints their ratio directly:

    [check] mxfp8 parity: mega rel_rms / torch rel_rms = 1.03x

A ratio near 1 says the mega kernel's error is normal for the format. That is
a far stronger statement than any absolute rel_rms, which mostly reflects the
format's own error floor (e4m3 keeps 3 mantissa bits, e2m1 keeps 1) and the
geometry. There is no in-tree NVFP4 grouped GEMM, so nvfp4 and fp8_fp4 get no
parity line and can only be read against BF16.

Each comparison is gated on the tolerance for its own pair, so a BF16-vs-BF16
line is never waved through by a loose quantized tolerance. The PASS/FAIL
metric is rel_rms; rel is reported but not gated, being a single-element max
that swings wildly under quantization. Neither catches subtle issues -- they
catch gross breakage, and a wrong weight layout or expert mapping lands near
1.0.

Measured on the default geometry (H=I=128, T=32, EP=4), rel_rms:

    vllm vs torch       0.0        bit-identical, all ranks, every precision
    mega bf16           ~4.7e-3
    mega mxfp8          ~6.6e-2
    mega nvfp4          ~2.5e-1

The quantized numbers are inflated by the toy geometry: error averages down
over the GEMM reduction dim, and H=I=128 gives it almost nothing to average
over. Expect materially better at model-sized H. What matters more than the
absolute value is agreement across ranks -- one rank far off while the others
agree means expert mapping, not numerics.

Example (EP=2 on one node):

  torchrun --nproc_per_node=2 scripts/local/benchmark_moe_vllm_vs_mega_bf16.py \\
      --expert-parallel-size 2 --local-tokens 32 --iters 20

  # vLLM only, on NVLS instead of NCCL:
  torchrun --nproc_per_node=2 scripts/local/benchmark_moe_vllm_vs_mega_bf16.py \\
      --backend vllm --vllm-dispatcher nvls

  # mega only:
  torchrun --nproc_per_node=2 scripts/local/benchmark_moe_vllm_vs_mega_bf16.py \\
      --backend flashinfer_mega

  # correctness instead of timing: every variant runs on the same weights and
  # input, compared against torch:bf16.
  torchrun --nproc_per_node=2 scripts/local/benchmark_moe_vllm_vs_mega_bf16.py \\
      --check

  # MXFP8: checks mega:mxfp8 and torch:mxfp8 against torch:bf16 and prints
  # their parity ratio; the timing run compares the same two implementations.
  torchrun --nproc_per_node=4 scripts/local/benchmark_moe_vllm_vs_mega_bf16.py \\
      --mega-precision mxfp8 --check
  torchrun --nproc_per_node=4 scripts/local/benchmark_moe_vllm_vs_mega_bf16.py \\
      --mega-precision mxfp8

  # nvfp4 has no in-tree counterpart, so it is BF16-relative only:
  torchrun --nproc_per_node=2 scripts/local/benchmark_moe_vllm_vs_mega_bf16.py \\
      --mega-precision nvfp4 --check

  # DeepSeek-V3 routed-MoE geometry, EP=4 (64 experts/rank, 5.25 GiB/rank bf16):
  torchrun --nproc_per_node=4 scripts/local/benchmark_moe_vllm_vs_mega_bf16.py \\
      --preset dsv3 --expert-parallel-size 4 \\
      --local-tokens 128 --mega-max-tokens 256 --mega-precision nvfp4
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
from megatron.core.parallel_state import (
    destroy_model_parallel,
    get_expert_model_parallel_group,
    initialize_model_parallel,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.token_dispatcher_inference import (
    MegaLocalPassthroughDispatcher,
    NCCLAllGatherDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_rank


# Max relative error tolerated against a BF16 oracle, per mega precision. The
# quantized kernels are compared against unquantized reference math, so the
# floor is set by the format, not by kernel quality: e4m3 keeps 3 mantissa bits
# and e2m1 keeps 1, hence the wide tolerances. rel_rms in the report is the
# metric that actually tracks quality; these gates only catch gross breakage
# (a wrong weight layout lands at rel ~1).
_DEFAULT_CHECK_TOL = {"bf16": 2e-2, "mxfp8": 1e-1, "nvfp4": 3e-1, "fp8_fp4": 3e-1}


# The correctness oracle, fixed: eager in-tree grouped GEMM, no quantization.
_REF_VARIANT = ("torch", "bf16")

# Precisions Megatron inference itself implements for the grouped MoE GEMM, by
# backend. There is no in-tree NVFP4: megatron/core/inference/quantization holds
# only MXFP8, so mega's nvfp4 and fp8_fp4 can only be read against BF16.
_MCORE_PRECISIONS = {"torch": ("bf16", "mxfp8"), "vllm": ("bf16",)}


# Two backends quantizing the same BF16 weights to the same format should
# agree far more closely than either agrees with unquantized math: the weight
# quantization error is common to both and largely cancels, leaving activation
# quantization and accumulation order. Provisional -- calibrate from the first
# matched run rather than trusting this number.
_MATCHED_PRECISION_TOL = 5e-2


def _select_variants(backend_filter: str, mega_precision: str) -> list[tuple[str, str]]:
    """Pick the (backend, precision) pairs to evaluate.

    Megatron's own quantized path is included whenever the mega kernel is
    running a format Megatron also implements, since that comparison is the
    only one that says whether mega's error is normal for the format.
    """
    variants: list[tuple[str, str]] = []
    if backend_filter in ("auto", "all", "vllm"):
        variants.append(("vllm", "bf16"))
    if backend_filter in ("auto", "all", "torch"):
        if mega_precision in _MCORE_PRECISIONS["torch"] and mega_precision != "bf16":
            variants.append(("torch", mega_precision))
        if backend_filter in ("all", "torch"):
            variants.append(("torch", "bf16"))
    if backend_filter in ("auto", "all", "flashinfer_mega"):
        variants.append(("flashinfer_mega", mega_precision))
    return variants


def _pair_tolerance(backend_precision: str, ref_precision: str, override: Optional[float]) -> float:
    """Tolerance for one backend-vs-reference comparison."""
    if override is not None:
        return override
    if backend_precision == ref_precision:
        return _DEFAULT_CHECK_TOL["bf16"] if backend_precision == "bf16" else _MATCHED_PRECISION_TOL
    # Mismatched formats: the comparison is bounded by whichever side is
    # coarser, so take the looser of the two gates.
    return max(_DEFAULT_CHECK_TOL[backend_precision], _DEFAULT_CHECK_TOL[ref_precision])


def _make_config(
    *,
    ep_size: int,
    backend: str,
    dispatcher: str,
    hidden: int,
    moe_ffn: int,
    mega_max_tokens: int,
    num_experts: int,
    topk: int,
    precision: str = "bf16",
    cpu_init: bool = True,
    mcore_mxfp8: bool = False,
) -> TransformerConfig:
    # Megatron-side MXFP8 (the 'torch' backend's own quantized path). This is
    # distinct from inference_mega_precision: config validation rejects the two
    # together, because the mega weight packer cannot read an MXFP8Tensor.
    #
    # Fused quant must be off for SwiGLU: the fused permute/activation+quantize
    # kernels are squared-ReLU only and _get_activation_func raises
    # NotImplementedError for SwiGLU. The unfused path quantizes to MXFP8 in a
    # separate launch before each GEMM, so both GEMMs are still MXFP8 -- only
    # the kernel fusion is lost, which costs launch overhead, not precision.
    fp8_kwargs = (
        dict(
            fp8="e4m3",
            fp8_recipe="mxfp8",
            fp8_param=True,
            inference_moe_disable_fused_quant_kernels=True,
        )
        if mcore_mxfp8
        else {}
    )
    return TransformerConfig(
        **fp8_kwargs,
        num_layers=1,
        hidden_size=hidden,
        ffn_hidden_size=2 * moe_ffn,
        num_attention_heads=4,
        num_query_groups=2,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=moe_ffn,
        moe_router_topk=topk,
        moe_router_score_function="softmax",
        # inference_optimized rejects anything else.
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_shared_expert_intermediate_size=None,
        # SwiGLU is the only activation the mega BF16 megakernel implements; the
        # vLLM path supports it too, so both backends run the same geometry.
        activation_func=F.silu,
        gated_linear_unit=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
        inference_grouped_gemm_backend=backend,
        inference_moe_token_dispatcher_type=dispatcher,
        inference_mega_precision=precision,
        inference_mega_max_tokens_per_rank=mega_max_tokens,
        attention_backend=AttnBackend.local,
        # TE allocates MXFP8 params on the GPU under fp8_model_init; CPU init
        # is not a supported combination. Harmless either way here, since the
        # shared BF16 weights overwrite whatever init produced.
        use_cpu_initialization=cpu_init and not mcore_mxfp8,
    )


def _allocate_dispatcher_buffers(config: TransformerConfig, max_tokens: int) -> None:
    ep_group = get_expert_model_parallel_group()
    if config.inference_grouped_gemm_backend == InferenceGroupedGemmBackend.FLASHINFER_MEGA:
        # Passthrough dispatcher: only the shared valid-tokens scalar is needed.
        MegaLocalPassthroughDispatcher.allocate_buffers()
        return
    if config.inference_moe_token_dispatcher_type == "nccl":
        NCCLAllGatherDispatcher.allocate_buffers()
    else:
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            NVLSAllGatherVDispatcher,
        )

        NVLSAllGatherVDispatcher.allocate_buffers(
            per_rank_worst_case_token_count=max_tokens,
            topk=config.moe_router_topk,
            hidden_size=config.hidden_size,
            ep_group=ep_group,
        )


def _build_layer(config: TransformerConfig, state_dict=None):
    """Build an inference MoE layer, optionally seeded with shared weights.

    ``state_dict`` must be applied before the first forward: both the
    concatenated-weight cache and the mega adapter's weight snapshot are built
    lazily on first call and would otherwise capture the random init.

    For a Megatron-side MXFP8 config the BF16 ``state_dict`` is loaded into
    fp8_param tensors and then converted, mirroring how production loads a
    BF16 checkpoint before quantizing (see megatron/inference/utils.py).
    """
    # TE only allocates MXFP8 parameters when the module is constructed inside
    # fp8_model_init, which get_fp8_context(is_init=True) supplies when
    # config.fp8_param is set (and a nullcontext otherwise). Constructing
    # outside it leaves the params BF16, and every later MXFP8 step then
    # silently does nothing -- see _assert_mxfp8_active.
    from megatron.core.fp8_utils import get_fp8_context

    with get_fp8_context(config, is_init=True):
        layer = get_inference_optimized_moe_spec()(config=config).cuda().eval()
    if state_dict is not None:
        layer.load_state_dict(state_dict)
    if config.fp8 and config.fp8_recipe == "mxfp8":
        from megatron.core.inference.quantization.utils import (
            quantize_model_to_mxfp8,
            resolve_mxfp8_backend,
        )

        quantize_model_to_mxfp8(
            layer, backend=resolve_mxfp8_backend(config.inference_grouped_gemm_backend)
        )
        _assert_mxfp8_active(layer)
    return layer


def _assert_mxfp8_active(layer) -> None:
    """Fail loudly if the expert weights are not actually MXFP8.

    InferenceGroupedMLP.forward picks its quantized path by duck-typing
    linear_fc1.weight0, not from the config: a non-MXFP8Tensor falls through
    to the BF16 grouped GEMM with no warning. Combined with
    quantize_model_to_mxfp8 being a no-op on non-TE-MXFP8 params, a misbuilt
    layer runs pure BF16 and reports exactly zero error against the BF16
    reference -- which looks like a passing test rather than a broken one.
    """
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    for name, module in layer.named_modules():
        weight = getattr(module, "weight0", None)
        if weight is None:
            continue
        if isinstance(weight, MXFP8Tensor) or isinstance(
            getattr(weight, "data", None), MXFP8Tensor
        ):
            return
        raise RuntimeError(
            f"MXFP8 requested but {name}.weight0 is {type(weight).__name__} "
            f"(data={type(getattr(weight, 'data', None)).__name__}). The layer "
            "would silently run BF16. Expected TE to allocate MXFP8 params "
            "under fp8_model_init (config.fp8_param) and "
            "quantize_model_to_mxfp8 to convert them to mcore MXFP8Tensor."
        )
    raise RuntimeError("MXFP8 requested but no expert weight0 was found to verify.")


def _run_correctness(
    *,
    ep_size: int,
    variants: list[tuple[str, str]],
    local_tokens: int,
    max_tokens: int,
    hidden_size: int,
    moe_ffn: int,
    mega_max_tokens: int,
    num_experts: int,
    topk: int,
    dispatcher: str,
    cpu_init: bool,
    tol: Optional[float],
) -> bool:
    """Compare every variant against the torch BF16 reference.

    The reference is always torch:bf16 -- the eager in-tree grouped GEMM with
    no quantization anywhere. Holding it fixed is what lets the quantized
    variants be read against each other: mega:mxfp8 and torch:mxfp8 measured
    against the same oracle are directly comparable error magnitudes.
    """
    rank = dist.get_rank()
    order = [_REF_VARIANT] + [v for v in variants if v != _REF_VARIANT]

    torch.manual_seed(1234 + rank)
    hidden_states = torch.randn(local_tokens, 1, hidden_size, device="cuda", dtype=torch.bfloat16)

    def build_config(backend: str, precision: str) -> TransformerConfig:
        return _make_config(
            ep_size=ep_size,
            backend=backend,
            dispatcher="nccl" if backend == "flashinfer_mega" else dispatcher,
            hidden=hidden_size,
            moe_ffn=moe_ffn,
            mega_max_tokens=mega_max_tokens,
            num_experts=num_experts,
            topk=topk,
            precision=precision if backend == "flashinfer_mega" else "bf16",
            cpu_init=cpu_init,
            mcore_mxfp8=backend == "torch" and precision == "mxfp8",
        )

    # Snapshot BF16 weights from a throwaway layer rather than from the
    # reference layer: torch:mxfp8's state_dict holds MXFP8Tensors that the
    # other variants cannot load. Every variant therefore starts from the same
    # BF16 weights and applies its own quantization, which is what makes the
    # error magnitudes comparable.
    seed_layer = _build_layer(build_config("torch", "bf16"))
    shared_state = {k: v.clone() for k, v in seed_layer.state_dict().items()}
    del seed_layer
    torch.cuda.empty_cache()

    outputs: dict[tuple[str, str], torch.Tensor] = {}
    for backend, precision in order:
        config = build_config(backend, precision)
        _allocate_dispatcher_buffers(config, max_tokens)
        layer = _build_layer(config, shared_state)
        with torch.no_grad(), InferenceMode.active():
            out, _ = layer(hidden_states.clone(), padding_mask=None)
        outputs[(backend, precision)] = out.detach().float()
        del layer
        # Variants are built one at a time but each holds a full expert stack
        # (~5.25 GiB/rank at DS-V3 scale, EP=4), plus the mega kernel's
        # preprocessed copy. Return it before building the next one.
        torch.cuda.empty_cache()

    ref = outputs[_REF_VARIANT]
    ref_scale = ref.abs().max().clamp_min(1e-6)
    ref_norm = ref.norm().clamp_min(1e-6)
    ok = True
    rel_rms_by_variant: dict[tuple[str, str], float] = {}
    for variant in order:
        if variant == _REF_VARIANT:
            continue
        backend, precision = variant
        delta = outputs[variant] - ref
        diff = delta.abs()
        max_abs = diff.max().item()
        rel = max_abs / ref_scale.item()
        # Whole-tensor relative error; unlike rel it is not set by a single
        # outlier element, so it is the number to watch across precisions.
        rel_rms = (delta.norm() / ref_norm).item()
        rel_rms_by_variant[variant] = rel_rms
        variant_tol = _pair_tolerance(precision, _REF_VARIANT[1], tol)
        # Gate on rel_rms, not rel: rel is a single-element max, so under
        # quantization it swings by tens of percent between ranks on identical
        # math and turns the gate into a coin flip. rel_rms is stable to a few
        # percent and still catches real damage -- one corrupted expert out of
        # E moves it by ~sqrt(1/E), far above any of these tolerances.
        passed = rel_rms <= variant_tol
        ok = ok and passed
        # Every rank reports: a mismatch may be confined to one EP rank.
        print(
            f"[check rank{rank}] {backend}:{precision} vs torch:bf16: "
            f"max_abs={max_abs:.3e} mean_abs={diff.mean().item():.3e} "
            f"rel={rel:.3e} rel_rms={rel_rms:.3e} tol={variant_tol:.1e} "
            f"-> {'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    _report_quantization_parity(rel_rms_by_variant, rank)

    flag = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(flag)
    all_ok = flag.item() == 0
    if rank == 0:
        print(f"[check] correctness: {'PASS' if all_ok else 'FAIL'} (all {ep_size} EP ranks)")
    return all_ok


def _report_quantization_parity(rel_rms: dict[tuple[str, str], float], rank: int) -> None:
    """Compare the mega kernel's error to Megatron's own path at the same format.

    Both are measured against the same BF16 reference, so their ratio answers
    the question the absolute numbers cannot: is the mega kernel's error the
    same order as the in-tree implementation of that format, or is something
    wrong with it? A ratio near 1 means the two quantize comparably.
    """
    for (backend, precision), value in sorted(rel_rms.items()):
        if backend != "flashinfer_mega":
            continue
        peer = rel_rms.get(("torch", precision))
        if peer is None:
            continue
        ratio = value / max(peer, 1e-12)
        worst = torch.tensor([ratio], device="cuda")
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"[check] {precision} parity: mega rel_rms / torch rel_rms = "
                f"{worst.item():.2f}x (worst rank; ~1 => same error level)"
            )


def _run_layer_benchmark(
    config: TransformerConfig,
    local_tokens: int,
    warmup: int,
    iters: int,
) -> float:
    layer = _build_layer(config)
    rank = get_pg_rank(get_expert_model_parallel_group())
    torch.manual_seed(42 + rank)
    # The layer's own router produces probs/routing_map from these hidden states.
    hidden_states = torch.randn(
        local_tokens, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )

    def _step():
        with torch.no_grad(), InferenceMode.active():
            out, _ = layer(hidden_states.clone(), padding_mask=None)
        return out

    for _ in range(warmup):
        _step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vLLM vs flashinfer_mega MoE")
    parser.add_argument("--expert-parallel-size", type=int, default=None)
    parser.add_argument("--local-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--backend",
        choices=("auto", "all", "vllm", "torch", "flashinfer_mega"),
        default="auto",
        help="Which backends to run. 'auto' is vllm:bf16 + mega, plus "
        "torch at the mega precision when Megatron implements that format "
        "(mxfp8 only). 'all' additionally includes torch:bf16 as a timed "
        "variant. In --check mode torch:bf16 always runs as the reference.",
    )
    parser.add_argument(
        "--vllm-dispatcher",
        choices=("nccl", "nvls"),
        default="nccl",
        help="Inference token dispatcher for the vLLM baseline. Ignored by the mega "
        "path, which always uses the local passthrough dispatcher.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare every variant against the torch:bf16 reference (shared "
        "weights, identical input) instead of timing.",
    )
    parser.add_argument(
        "--mega-precision",
        choices=tuple(_DEFAULT_CHECK_TOL),
        default="bf16",
        help="Numerics of the mega kernel. Weights are always handed to FlashInfer "
        "as BF16; the quantized configs quantize weights at layer construction "
        "and activations in-kernel, so no calibration or external scales are "
        "needed. 'fp8_fp4' additionally requires the deep_gemm package.",
    )
    parser.add_argument(
        "--check-tol",
        type=float,
        default=None,
        help="Override the --check tolerance on rel_rms for every comparison. "
        "By default each backend-vs-reference pair gets its own: matching "
        f"quantized formats {_MATCHED_PRECISION_TOL:g}, otherwise the looser of "
        + ", ".join(f"{k}={v:g}" for k, v in _DEFAULT_CHECK_TOL.items()),
    )
    parser.add_argument("--mega-max-tokens", type=int, default=128)
    # Shape flags default to None so an explicit value can be told apart from
    # an unset one, letting it win over --preset (resolved below).
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument(
        "--moe-ffn-hidden-size",
        type=int,
        default=None,
        help="Post-SwiGLU expert width. Alignment is per mega precision "
        "(bf16 %%64, mxfp8 %%32, nvfp4 %%16, fp8_fp4 %%32). Default 128.",
    )
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument(
        "--gpu-init",
        action="store_true",
        help="Initialize expert weights directly on the GPU. CPU init is the "
        "default but costs minutes at model scale (DS-V3 is ~2.8B expert "
        "params per rank at EP=4); implied by --preset dsv3.",
    )
    parser.add_argument(
        "--preset",
        choices=("dsv3",),
        help="Overwrite the shape flags with a named model geometry. 'dsv3' is "
        "DeepSeek-V3's routed MoE: H=7168, I=2048, 256 experts, topk=8. Only "
        "the shapes are copied -- the router here stays softmax rather than "
        "V3's sigmoid group-limited routing, which changes token distribution "
        "across experts but not the GEMM shapes.",
    )
    args = parser.parse_args()

    # Precedence: explicit flag > --preset > built-in default. Without this an
    # explicit --hidden-size alongside --preset would be silently discarded.
    shape_defaults = {
        "hidden_size": 128,
        "moe_ffn_hidden_size": 128,
        "num_experts": 8,
        "topk": 2,
    }
    preset_shapes = {
        "dsv3": {
            "hidden_size": 7168,
            "moe_ffn_hidden_size": 2048,
            "num_experts": 256,
            "topk": 8,
        }
    }.get(args.preset, {})
    for name, fallback in shape_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, preset_shapes.get(name, fallback))
    if args.preset == "dsv3":
        # CPU init of ~2.75B expert params per rank would dominate the run.
        args.gpu_init = True

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    # Must precede init_process_group: NCCL rejects several ranks sharing a device.
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl")
    world = dist.get_world_size()
    ep_size = args.expert_parallel_size or world
    if world % ep_size != 0:
        raise SystemExit(f"world_size {world} must be divisible by EP {ep_size}")

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
    )

    max_tokens = max(args.local_tokens * 4, args.mega_max_tokens)
    results: dict[tuple[str, str], float] = {}
    variants = _select_variants(args.backend, args.mega_precision)

    if args.check:
        from megatron.core.inference.moe.mega._deps import require_flashinfer_moe_ep

        require_flashinfer_moe_ep()
        all_ok = _run_correctness(
            ep_size=ep_size,
            variants=variants,
            local_tokens=args.local_tokens,
            max_tokens=max_tokens,
            hidden_size=args.hidden_size,
            moe_ffn=args.moe_ffn_hidden_size,
            mega_max_tokens=args.mega_max_tokens,
            num_experts=args.num_experts,
            topk=args.topk,
            dispatcher=args.vllm_dispatcher,
            cpu_init=not args.gpu_init,
            tol=args.check_tol,
        )
        destroy_model_parallel()
        dist.destroy_process_group()
        raise SystemExit(0 if all_ok else 1)

    for backend, precision in variants:
        # The mega path bypasses Megatron's EP gather entirely, so its dispatcher
        # choice is irrelevant; pin it to nccl to avoid allocating NVLS buffers.
        dispatcher = "nccl" if backend == "flashinfer_mega" else args.vllm_dispatcher
        if backend == "flashinfer_mega":
            try:
                from megatron.core.inference.moe.mega._deps import require_flashinfer_moe_ep

                require_flashinfer_moe_ep()
            except RuntimeError as exc:
                if dist.get_rank() == 0:
                    print(f"Skipping flashinfer_mega: {exc}")
                continue

        config = _make_config(
            ep_size=ep_size,
            backend=backend,
            dispatcher=dispatcher,
            hidden=args.hidden_size,
            moe_ffn=args.moe_ffn_hidden_size,
            mega_max_tokens=args.mega_max_tokens,
            num_experts=args.num_experts,
            topk=args.topk,
            precision=precision if backend == "flashinfer_mega" else "bf16",
            cpu_init=not args.gpu_init,
            mcore_mxfp8=backend == "torch" and precision == "mxfp8",
        )
        if args.local_tokens > config.inference_mega_max_tokens_per_rank:
            raise SystemExit(
                f"--local-tokens {args.local_tokens} exceeds mega cap "
                f"{config.inference_mega_max_tokens_per_rank}"
            )

        _allocate_dispatcher_buffers(config, max_tokens)
        avg_s = _run_layer_benchmark(
            config, args.local_tokens, args.warmup, args.iters
        )
        results[(backend, precision)] = avg_s
        if dist.get_rank() == 0:
            tok_per_s = args.local_tokens / avg_s
            print(
                f"[{backend}:{precision}] "
                f"EP={ep_size} local_tokens={args.local_tokens} "
                f"avg={avg_s * 1000:.3f} ms/step  ~{tok_per_s:.1f} local tok/s/rank"
            )

    mega_key = ("flashinfer_mega", args.mega_precision)
    if dist.get_rank() == 0 and mega_key in results:
        # Against the same format first: that is the like-for-like speedup.
        # The BF16 baselines are the "what does this cost today" comparison.
        baselines = sorted(results, key=lambda v: v[1] != args.mega_precision)
        for baseline in baselines:
            if baseline == mega_key:
                continue
            speedup = results[baseline] / results[mega_key]
            print(
                f"mega:{args.mega_precision} vs {baseline[0]}:{baseline[1]} "
                f"time ratio ({baseline[0]}/mega): {speedup:.3f}x  (>1 => mega faster)"
            )

    destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
