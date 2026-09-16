# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lazy wrapper around FlashInfer ``MoEEpMegaLayer`` for one MoE block."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from megatron.core.inference.moe.mega._deps import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpMegaLayer,
    MoEEpTensors,
    require_flashinfer_moe_ep,
)
from megatron.core.inference.moe.mega.registry import build_megakernel_config
from megatron.core.inference.moe.mega.weights import megatron_grouped_weights_to_moe_pack
from megatron.core.utils import get_pg_rank, get_pg_size

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


class MegatronMegaMoEAdapter:
    """One FlashInfer mega layer per Megatron ``InferenceGroupedMLP``.

    In inference, ``MoEEpMegaLayer`` transforms the canonical weight pack at
    construction and releases the source tensors, so the expert weights are
    snapshotted on the first forward. Refitting Megatron's expert parameters
    afterwards does not reach the kernel; the adapter must be rebuilt instead.

    ``owns_transformed_weights`` inverts that for training, where the optimizer
    rewrites the parameters every step. The caller then supplies weights that
    are already in the kernel's layout and keeps rewriting them in place, and
    FlashInfer's own preprocessing is bypassed so it cannot snapshot anything.
    See :mod:`megatron.core.inference.moe.mega.training_weights`.
    """

    def __init__(
        self,
        config: "TransformerConfig",
        ep_group: torch.distributed.ProcessGroup,
        owns_transformed_weights: bool = False,
    ) -> None:
        require_flashinfer_moe_ep()
        self._config = config
        self._ep_group = ep_group
        self._layer: Optional[MoEEpMegaLayer] = None
        self._warmed_up = False
        self._owns_transformed_weights = owns_transformed_weights

    def _fleet_params(self) -> FleetParams:
        # dtype_bytes/algorithm/layout are split-transport fields the mega path
        # ignores; they are left at their defaults.
        return FleetParams(
            num_experts=self._config.num_moe_experts,
            max_tokens_per_rank=self._config.inference_mega_max_tokens_per_rank,
            token_hidden_size=self._config.hidden_size,
        )

    def _bootstrap(self) -> BootstrapConfig:
        return BootstrapConfig(
            world_size=get_pg_size(self._ep_group),
            rank=get_pg_rank(self._ep_group),
            process_group=self._ep_group,
            device=torch.cuda.current_device(),
        )

    def _ensure_layer(
        self,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ) -> MoEEpMegaLayer:
        if self._layer is not None:
            return self._layer
        # Construction runs collective symmetric-heap bootstrap and weight
        # preprocessing, neither of which can be captured.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "flashinfer_mega layer construction cannot run during CUDA graph "
                "capture. Run one eager forward on all EP ranks before capturing."
            )
        megakernel = build_megakernel_config(self._config)
        # quantize_input must stay True: the megakernels have no pre-quantized
        # activation path and reject quantize_input=False. The quantized
        # precisions derive activation scales in-kernel, so no calibration data
        # is needed here.
        if self._owns_transformed_weights:
            # fc1/fc2 are already K-major kernel layout and the caller mutates
            # them in place every step, so there is nothing to preprocess and
            # nothing may be released. FlashInfer validates the layout here,
            # which is what catches a repack that does not match its contract.
            weights = None
            mega_config = MegaConfig(
                megakernel=megakernel,
                preprocess_weights=False,
                transformed_weights=((fc1_weight, None), (fc2_weight, None)),
            )
        else:
            weights = megatron_grouped_weights_to_moe_pack(fc1_weight, fc2_weight)
            mega_config = MegaConfig(megakernel=megakernel, preprocess_weights=True)
        self._layer = MoEEpMegaLayer(
            self._bootstrap(),
            self._fleet_params(),
            weights,
            mega_config,
        )
        return self._layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Run local-token mega MoE; EP communication is inside the kernel."""
        num_tokens = hidden_states.shape[0]
        max_cap = self._config.inference_mega_max_tokens_per_rank
        if num_tokens > max_cap:
            raise ValueError(
                f"Mega MoE received {num_tokens} local tokens, exceeding "
                f"inference_mega_max_tokens_per_rank={max_cap}. "
                "Increase the cap or reduce batch tokens per EP rank."
            )

        layer = self._ensure_layer(fc1_weight, fc2_weight)
        if not self._warmed_up:
            # warmup() is an EP collective that forces workspace allocation and
            # CuTeDSL compilation; both are illegal under capture.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "flashinfer_mega requires warmup() on all EP ranks before CUDA "
                    "graph capture. Run one eager forward first."
                )
            layer.warmup()
            self._warmed_up = True

        tensors = MoEEpTensors(
            hidden_states=hidden_states.contiguous(),
            topk_ids=routing_map,
            topk_weights=probs.to(torch.float32),
        )
        return layer.forward(tensors)
