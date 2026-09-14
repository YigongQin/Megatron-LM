# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Precision registry for FlashInfer mega MoE kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import flashinfer.moe_ep.backends  # noqa: F401 — register mega kernels
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl import (
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
    )

    _HAVE_MEGA_CONFIG = True
except ImportError:
    _HAVE_MEGA_CONFIG = False
    Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class MegaPrecisionSpec:
    """Maps a precision name to a FlashInfer megakernel config factory."""

    name: str
    build_config: Callable[["TransformerConfig"], Any]


def _build_bf16_cutedsl_config(config: "TransformerConfig") -> Any:
    if not _HAVE_MEGA_CONFIG:
        raise RuntimeError(
            "FlashInfer moe_ep sm100 BF16 mega kernel is not available. It requires "
            "flashinfer built from a source tree containing "
            "flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_bf16_bf16_cutedsl "
            "(unreleased as of 0.6.18; targeted for 0.7.0)."
        )
    # The kernel derives the gate+up width internally as 2 * intermediate_size,
    # so intermediate_size is the post-SwiGLU width.
    # gate_up_clamp/activation_clamp are left unset: FlashInfer hard-clamps the
    # FC1 output, whereas activation_func_tanh_clamp_scale is a soft tanh clamp
    # that replaces the swish gate entirely (SiTU-GLU). Config validation
    # rejects the clamp for this backend rather than silently mismapping it.
    return Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=config.moe_ffn_hidden_size, top_k=config.moe_router_topk
    )


# Extend this table when adding mxfp8 / nvfp4 mega paths.
MEGA_PRECISION_REGISTRY: dict[str, MegaPrecisionSpec] = {
    "bf16": MegaPrecisionSpec(name="bf16", build_config=_build_bf16_cutedsl_config),
}


def get_mega_precision_spec(precision: str) -> MegaPrecisionSpec:
    try:
        return MEGA_PRECISION_REGISTRY[precision]
    except KeyError as exc:
        supported = ", ".join(sorted(MEGA_PRECISION_REGISTRY))
        raise ValueError(
            f"Unknown inference_mega_precision={precision!r}; supported: {supported}"
        ) from exc


def build_megakernel_config(config: "TransformerConfig") -> Any:
    """Build the FlashInfer megakernel config object for the model."""
    spec = get_mega_precision_spec(config.inference_mega_precision)
    return spec.build_config(config)
