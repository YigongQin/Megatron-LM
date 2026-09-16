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

Generation reuses the packing for a different reason. Static weights would be
fine to hand FlashInfer once, but RL refits them between generations, and a
layer that let FlashInfer preprocess would keep serving the weights it
snapshotted at construction -- silently, since nothing reads the parameters
again. Owning the buffer instead means a refit can rewrite it in place and the
next forward sees new weights, with no teardown, no EP collective and no
recompile. Memory is unchanged, because FlashInfer holds exactly one
transformed copy either way. That path is BF16-only; see
:class:`MegaKernelWeightBuffer`.
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


class MegaKernelWeightBuffer:
    """A pair of kernel-layout expert weight buffers, and the repack into them.

    The buffers are in canonical Megatron layout and the kernel consumes
    transposed views of them, so rewriting them in place is visible to a
    FlashInfer layer that was handed those views once.

    Two callers with opposite lifetimes share this. The training forward wants
    one buffer for the whole model, repacked as execution walks the layers
    (:class:`MegaTrainingWeightScratch`). Generation wants one buffer per layer,
    persistent, rewritten only when the weights actually change -- which under
    RL means after a refit. Both need the packing to be byte-identical to
    FlashInfer's own ``preprocess_mega_weights``, which is what
    ``tests/unit_tests/inference/test_mega_training_weights.py`` pins.

    BF16 only. For the quantized mega precisions FlashInfer's preprocessing also
    quantizes the weights, and this reproduces only its interleave and
    transpose, so a quantized layer must let FlashInfer preprocess instead.
    """

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

    def views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(fc1, fc2)`` K-major views the kernel consumes.

        Stable across repacks: the transposes alias the buffers, so a layer
        handed these once keeps seeing whatever was last packed into them.
        """
        return self._w13.transpose(1, 2), self._w2.transpose(1, 2)

    def repack(
        self, fc1_weights: list[torch.Tensor], fc2_weights: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild the kernel layout from one layer's live expert weights.

        Args:
            fc1_weights: per-expert ``[2 * intermediate, hidden]`` gate-then-up
                weights, in local expert order.
            fc2_weights: per-expert ``[hidden, intermediate]`` weights.

        Returns:
            The ``(fc1, fc2)`` K-major views, as :meth:`views`.
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

        return self.views()


class MegaTrainingWeightScratch(MegaKernelWeightBuffer):
    """One process-wide buffer pair, shared by every MoE layer in training.

    :meth:`repack` overwrites the buffers from one layer's parameters and
    returns the views the kernel consumes; they stay valid only until the next
    :meth:`repack`, hence the ownership check.
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
        super().__init__(num_local_experts, hidden_size, intermediate_size, dtype, device)
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
        """Repack for ``owner``, recording it so a stale read can be reported.

        Returns views valid only until the next call, unlike the base class,
        where each layer has its own buffer.
        """
        views = super().repack(fc1_weights, fc2_weights)
        self._owner = owner
        return views

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
