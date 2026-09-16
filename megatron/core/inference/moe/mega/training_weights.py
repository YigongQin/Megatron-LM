# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kernel-layout expert weights for the mega MoE training forward.

The inference path can hand FlashInfer the expert weights once and forget
them: they never change, so ``MoEEpMegaLayer`` transforms them at construction
and releases the source. Training cannot do that. The optimizer rewrites the
expert parameters every step, and the kernel layout is not the parameter
layout, so the kernel's copy has to be rebuilt from the live parameters before
every forward.

Two constraints shape this module.

Megatron's DDP owns parameter storage. It repoints ``param.data`` at views
into its own contiguous buffer and asserts the incoming data is not already a
view, so the inference trick of aliasing parameters onto one big tensor
(``InferenceGroupedMLP._build_concatenated_weights``) cannot be used while DDP
is active. The kernel's weights therefore have to be a genuine second copy,
gathered from the parameters rather than aliased onto them.

A full second copy per layer would roughly double expert-weight memory, which
is the dominant term for an MoE model. Instead one scratch buffer is shared by
every MoE layer and repacked as execution walks them, so the cost is one
layer's worth of expert weights for the whole model. That makes the buffer
single-use: its contents are only valid for the layer that most recently
repacked it, and only until the next repack. The mega kernel reads it
synchronously on the same stream inside that layer's forward, which is what
makes the sharing sound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

# Gate/up interleave granularity required by the sm100 BF16 CuTeDSL mega
# kernel. Must match flashinfer.moe_ep's own preprocessing, which interleaves
# gate and up rows in blocks of this size before transposing to K-major.
_GATE_UP_INTERLEAVE = 32


class MegaTrainingWeightScratch:
    """One process-wide pair of kernel-layout expert weight buffers.

    Shared by every MoE layer. :meth:`repack` overwrites the buffers from one
    layer's parameters and returns the transposed views the kernel consumes;
    the returned views stay valid only until the next :meth:`repack`.
    """

    _instance: Optional["MegaTrainingWeightScratch"] = None

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._key = (num_local_experts, hidden_size, intermediate_size, dtype, device)
        # Canonical Megatron layout. The kernel views are transposes of these,
        # so writing here in place updates what the kernel sees.
        self._w13 = torch.empty(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        self._w2 = torch.empty(
            num_local_experts, hidden_size, intermediate_size, dtype=dtype, device=device
        )
        self._owner: Optional[int] = None

    @classmethod
    def get(
        cls,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "MegaTrainingWeightScratch":
        """Return the shared scratch, allocating it on first use.

        Every MoE layer in a model has the same expert geometry, so one buffer
        serves all of them. A second geometry in the same process is rejected
        rather than silently allocating another multi-GiB buffer.
        """
        key = (num_local_experts, hidden_size, intermediate_size, dtype, device)
        if cls._instance is None:
            cls._instance = cls(
                num_local_experts, hidden_size, intermediate_size, dtype, device
            )
        elif cls._instance._key != key:
            raise RuntimeError(
                "mega training weight scratch is shared across MoE layers and was "
                f"allocated for {cls._instance._key}, but a layer requested {key}. "
                "Mixed expert geometries in one process are not supported."
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the shared buffers. For teardown between tests."""
        cls._instance = None

    def repack(
        self,
        fc1_weights: list[torch.Tensor],
        fc2_weights: list[torch.Tensor],
        owner: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild the kernel layout from one layer's live expert parameters.

        Args:
            fc1_weights: per-expert ``[2 * intermediate, hidden]`` gate-then-up
                weights, in local expert order.
            fc2_weights: per-expert ``[hidden, intermediate]`` weights.
            owner: identity of the calling layer, recorded so a stale read can
                be reported rather than silently returning another layer's
                weights.

        Returns:
            The ``(fc1, fc2)`` K-major views the mega kernel consumes. Valid
            only until the next call.
        """
        num_local_experts, hidden_size, intermediate_size, _, _ = self._key
        if len(fc1_weights) != num_local_experts or len(fc2_weights) != num_local_experts:
            raise ValueError(
                f"expected {num_local_experts} expert weights, got "
                f"{len(fc1_weights)} fc1 and {len(fc2_weights)} fc2"
            )

        blocks = intermediate_size // _GATE_UP_INTERLEAVE
        # [E, I/32, 2, 32, H]: index 0 of the size-2 axis holds gate blocks and
        # index 1 holds up blocks, which is the interleave the kernel expects.
        interleaved = self._w13.view(
            num_local_experts, blocks, 2, _GATE_UP_INTERLEAVE, hidden_size
        )
        for expert, (fc1, fc2) in enumerate(zip(fc1_weights, fc2_weights)):
            gate, up = fc1[:intermediate_size], fc1[intermediate_size:]
            interleaved[expert, :, 0].copy_(
                gate.view(blocks, _GATE_UP_INTERLEAVE, hidden_size)
            )
            interleaved[expert, :, 1].copy_(up.view(blocks, _GATE_UP_INTERLEAVE, hidden_size))
            self._w2[expert].copy_(fc2)

        self._owner = owner
        return self._w13.transpose(1, 2), self._w2.transpose(1, 2)

    def assert_owned_by(self, owner: int) -> None:
        """Fail loudly if another layer repacked the scratch since we did.

        Training against a stale buffer looks like a broken learning rate
        rather than an error, so the invariant is checked instead of assumed.
        """
        if self._owner != owner:
            raise RuntimeError(
                "mega training weight scratch was repacked by another MoE layer "
                f"(owner={self._owner}, expected {owner}). The scratch is only "
                "valid for the layer that most recently repacked it; concurrent "
                "or reordered MoE layer execution is not supported."
            )


def kernel_layout_from_parameters(
    config: "TransformerConfig",
    fc1_weights: list[torch.Tensor],
    fc2_weights: list[torch.Tensor],
    owner: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack one layer's expert parameters into the shared kernel-layout scratch."""
    num_local_experts = len(fc1_weights)
    scratch = MegaTrainingWeightScratch.get(
        num_local_experts=num_local_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_ffn_hidden_size,
        dtype=fc1_weights[0].dtype,
        device=fc1_weights[0].device,
    )
    return scratch.repack(fc1_weights, fc2_weights, owner=owner)
