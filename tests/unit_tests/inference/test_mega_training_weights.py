# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the mega training-forward weight scratch and its config gating."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.moe.mega.training_weights import (
    MegaTrainingWeightScratch,
    reset_training_scratches,
    kernel_layout_from_parameters,
    mxfp8_kernel_weights_from_views,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

_INTERLEAVE = 32


def _config(**overrides):
    base = dict(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        num_moe_experts=8,
        moe_ffn_hidden_size=128,
        moe_router_topk=2,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        activation_func=F.silu,
        gated_linear_unit=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="inference_optimized",
        inference_grouped_gemm_backend="flashinfer_mega",
        inference_mega_max_tokens_per_rank=64,
        attention_backend=AttnBackend.local,
        use_cpu_initialization=True,
        # moe_mega_training_forward requires MoE recompute.
        recompute_granularity="selective",
        recompute_modules=["moe"],
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _reference_kernel_layout(fc1_weights, fc2_weights, intermediate_size):
    """The layout the kernel actually expects, taken from flashinfer itself.

    Deferring to upstream is the point: a repack that merely agrees with our own
    reimplementation of the interleave would still be wrong if we misread the
    kernel's contract. Falls back to a local implementation only so the test
    remains useful where flashinfer is not installed.
    """
    w13 = torch.stack([w.clone() for w in fc1_weights])
    w2 = torch.stack([w.clone() for w in fc2_weights])
    try:
        from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights import (
            _interleave_gate_up_32,
        )

        return _interleave_gate_up_32(w13, intermediate_size).transpose(1, 2), w2.transpose(1, 2)
    except ImportError:
        hidden = w13.shape[2]
        blocks = intermediate_size // _INTERLEAVE
        gate, up = w13[:, :intermediate_size], w13[:, intermediate_size:]
        out = torch.empty_like(w13)
        view = out.view(w13.shape[0], blocks, 2, _INTERLEAVE, hidden)
        view[:, :, 0].copy_(gate.view(w13.shape[0], blocks, _INTERLEAVE, hidden))
        view[:, :, 1].copy_(up.view(w13.shape[0], blocks, _INTERLEAVE, hidden))
        return out.transpose(1, 2), w2.transpose(1, 2)


def _weights(num_experts, hidden, intermediate, dtype=torch.bfloat16):
    fc1 = [torch.randn(2 * intermediate, hidden, dtype=dtype) for _ in range(num_experts)]
    fc2 = [torch.randn(hidden, intermediate, dtype=dtype) for _ in range(num_experts)]
    return fc1, fc2


@pytest.fixture(autouse=True)
def _reset_scratch():
    reset_training_scratches()
    yield
    reset_training_scratches()


@pytest.mark.internal
class TestMegaTrainingWeightScratch:
    """The repack is the correctness-critical part: a wrong layout silently trains."""

    def test_repack_matches_reference_layout(self):
        config = _config()
        fc1, fc2 = _weights(4, config.hidden_size, config.moe_ffn_hidden_size)
        got_fc1, got_fc2 = kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        want_fc1, want_fc2 = _reference_kernel_layout(fc1, fc2, config.moe_ffn_hidden_size)
        assert torch.equal(got_fc1, want_fc1)
        assert torch.equal(got_fc2, want_fc2)

    def test_kernel_views_are_k_major(self):
        config = _config()
        fc1, fc2 = _weights(4, config.hidden_size, config.moe_ffn_hidden_size)
        got_fc1, got_fc2 = kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        assert got_fc1.shape == (4, config.hidden_size, 2 * config.moe_ffn_hidden_size)
        assert got_fc2.shape == (4, config.moe_ffn_hidden_size, config.hidden_size)

    def test_repack_picks_up_parameter_updates(self):
        """The whole reason this module exists: no snapshotting of stale weights."""
        config = _config()
        fc1, fc2 = _weights(4, config.hidden_size, config.moe_ffn_hidden_size)
        kernel_fc1, _ = kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        before = kernel_fc1.clone()

        # Stand in for an optimizer step.
        for w in fc1:
            w.add_(1.0)
        kernel_fc1, _ = kernel_layout_from_parameters(config, fc1, fc2, owner=1)

        assert not torch.equal(kernel_fc1, before)
        want_fc1, _ = _reference_kernel_layout(fc1, fc2, config.moe_ffn_hidden_size)
        assert torch.equal(kernel_fc1, want_fc1)

    def test_scratch_is_shared_between_layers(self):
        config = _config()
        fc1, fc2 = _weights(4, config.hidden_size, config.moe_ffn_hidden_size)
        first, _ = kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        second, _ = kernel_layout_from_parameters(config, fc1, fc2, owner=2)
        # Same storage: one buffer serves every MoE layer.
        assert first.data_ptr() == second.data_ptr()

    def test_stale_owner_is_rejected(self):
        """A second layer repacking before the first reads must not pass silently."""
        config = _config()
        fc1, fc2 = _weights(4, config.hidden_size, config.moe_ffn_hidden_size)
        kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        scratch = MegaTrainingWeightScratch.get(
            num_local_experts=4,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_ffn_hidden_size,
            dtype=torch.bfloat16,
            device=fc1[0].device,
        )
        scratch.assert_owned_by(1)
        kernel_layout_from_parameters(config, fc1, fc2, owner=2)
        with pytest.raises(RuntimeError, match="repacked by another MoE layer"):
            scratch.assert_owned_by(1)

    def test_mixed_expert_geometry_is_rejected(self):
        config = _config()
        fc1, fc2 = _weights(4, config.hidden_size, config.moe_ffn_hidden_size)
        kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        fc1_other, fc2_other = _weights(2, config.hidden_size, config.moe_ffn_hidden_size)
        with pytest.raises(RuntimeError, match="Mixed expert geometries"):
            kernel_layout_from_parameters(config, fc1_other, fc2_other, owner=2)

    def test_upstream_validator_accepts_repacked_views(self):
        """The repack hands the kernel transposed views, not contiguous tensors."""
        validate = pytest.importorskip(
            "flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.weights"
        ).validate_transformed_mega_weights
        config = _config()
        num_local_experts = 4
        fc1, fc2 = _weights(num_local_experts, config.hidden_size, config.moe_ffn_hidden_size)
        kernel_fc1, kernel_fc2 = kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        validate(
            ((kernel_fc1, None), (kernel_fc2, None)),
            intermediate_size=config.moe_ffn_hidden_size,
            hidden_size=config.hidden_size,
            world_size=1,
            num_experts=num_local_experts,
        )

    def test_expert_count_mismatch_is_rejected(self):
        config = _config()
        fc1, fc2 = _weights(4, config.hidden_size, config.moe_ffn_hidden_size)
        with pytest.raises(ValueError, match="expected 4 expert weights"):
            kernel_layout_from_parameters(config, fc1, fc2[:3], owner=1)


def _mxfp8_weights_module():
    """FlashInfer's mxfp8 mega weight module, or skip."""
    return pytest.importorskip(
        "flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights"
    )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP8 quantization is a device kernel")
class TestMxfp8KernelWeights:
    """Whether we can produce the quantized bytes FlashInfer would have produced.

    Refit is what forces the question. Generation owns the kernel's weight
    buffers so that a refit can rewrite them in place, which means our code --
    not ``preprocess_mega_weights`` -- decides the quantized bytes the kernel
    reads. Anything short of byte-equality makes the owned path a slightly
    different model from the preprocessed one, indistinguishable from a real
    train/generation gap, so these compare bytes rather than tolerances.
    """

    EXPERTS = 4

    def _quantized(self, config):
        """Our kernel-ready MXFP8 weights, and the bf16 weights they came from."""
        fc1, fc2 = _weights(self.EXPERTS, config.hidden_size, config.moe_ffn_hidden_size)
        fc1 = [w.cuda() for w in fc1]
        fc2 = [w.cuda() for w in fc2]
        views = kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        return mxfp8_kernel_weights_from_views(*views), (fc1, fc2)

    @staticmethod
    def _reference(config, fc1, fc2):
        """The same weights through FlashInfer's own preprocessing."""
        from flashinfer.moe_ep.weights import MoEWeightPack

        pack = MoEWeightPack(w13=torch.stack(fc1), w2=torch.stack(fc2))
        return _mxfp8_weights_module().preprocess_mega_weights(
            pack,
            intermediate_size=config.moe_ffn_hidden_size,
            hidden_size=config.hidden_size,
        )

    def test_matches_flashinfer_preprocessing(self):
        _mxfp8_weights_module()
        config = _config()
        got, (fc1, fc2) = self._quantized(config)
        want = self._reference(config, fc1, fc2)

        for label, mine, theirs in zip(("fc1", "fc2"), got, want):
            for name, ours, upstream in (
                ("data", mine[0], theirs[0]),
                ("scale", mine[1], theirs[1]),
            ):
                assert ours.shape == upstream.shape, (
                    f"{label} {name}: {tuple(ours.shape)} != {tuple(upstream.shape)}"
                )
                assert ours.dtype == upstream.dtype, f"{label} {name} dtype"
                # Compared as bytes: torch.equal has no fp8 kernel, and bytes are
                # the only thing the kernel actually reads.
                mismatched = (ours.view(torch.uint8) != upstream.view(torch.uint8)).sum()
                assert mismatched == 0, (
                    f"{label} {name}: {int(mismatched)} of {ours.numel()} bytes differ "
                    "from FlashInfer's preprocessing"
                )

    def test_upstream_validator_accepts_our_weights(self):
        """The kernel's own admission check for ``preprocess_weights=False``."""
        validate = _mxfp8_weights_module().validate_transformed_mega_weights
        config = _config()
        got, _ = self._quantized(config)
        validate(
            got,
            intermediate_size=config.moe_ffn_hidden_size,
            hidden_size=config.hidden_size,
            world_size=1,
            num_experts=self.EXPERTS,
        )

    def test_quantization_follows_parameter_updates(self):
        """A refit that changed nothing in the quantized bytes would be invisible."""
        _mxfp8_weights_module()
        config = _config()
        got, (fc1, fc2) = self._quantized(config)
        before = got[0][0].view(torch.uint8).clone()

        for w in fc1:
            w.add_(1.0)
        views = kernel_layout_from_parameters(config, fc1, fc2, owner=1)
        after = mxfp8_kernel_weights_from_views(*views)

        assert not torch.equal(after[0][0].view(torch.uint8), before)
        want = self._reference(config, fc1, fc2)
        assert torch.equal(
            after[0][0].view(torch.uint8), want[0][0].view(torch.uint8)
        ), "requantization diverged from FlashInfer after a parameter update"


@pytest.mark.internal
class TestMegaTrainingForwardConfig:
    def test_accepts_valid_config(self):
        config = _config(moe_mega_training_forward=True)
        assert config.moe_mega_training_forward

    def test_defaults_off(self):
        assert not _config().moe_mega_training_forward

    def test_requires_mega_backend(self):
        with pytest.raises(ValueError, match="requires inference_grouped_gemm_backend"):
            _config(
                moe_mega_training_forward=True,
                inference_grouped_gemm_backend="torch",
            )

    def test_requires_bf16_precision(self):
        with pytest.raises(ValueError, match="straight-through estimator"):
            _config(moe_mega_training_forward=True, inference_mega_precision="nvfp4")

    def test_straight_through_admits_mxfp8(self):
        """The opt-in exists so the gradient trade is stated, not to be a gate."""
        config = _config(
            moe_mega_training_forward=True,
            inference_mega_precision="mxfp8",
            moe_mega_training_straight_through=True,
        )
        assert config.moe_mega_training_straight_through

    def test_straight_through_still_rejects_a_precision_without_a_packer(self):
        """nvfp4 cannot rebuild its kernel weights, so the opt-in cannot save it."""
        with pytest.raises(ValueError, match="only bf16 and mxfp8 can be rebuilt"):
            _config(
                moe_mega_training_forward=True,
                inference_mega_precision="nvfp4",
                moe_mega_training_straight_through=True,
            )

    def test_straight_through_alone_is_rejected(self):
        """Ignoring it would read as having opted into something that is not running."""
        with pytest.raises(ValueError, match="only relaxes a restriction"):
            _config(moe_mega_training_straight_through=True)

    def test_requires_moe_recompute(self):
        with pytest.raises(ValueError, match="must come from a recompute pass"):
            _config(
                moe_mega_training_forward=True,
                recompute_granularity=None,
                recompute_modules=None,
            )

    def test_requires_moe_in_recompute_modules(self):
        with pytest.raises(ValueError, match="must come from a recompute pass"):
            _config(moe_mega_training_forward=True, recompute_modules=["core_attn"])

    def test_rejects_local_cuda_graphs(self):
        """Would silently leave the experts with no backward graph."""
        with pytest.raises(ValueError, match="cuda_graph_impl='local'"):
            _config(moe_mega_training_forward=True, cuda_graph_impl="local")

    def test_rejects_ep_comm_overlap(self):
        with pytest.raises(ValueError, match="overlap_moe_expert_parallel_comm"):
            _config(moe_mega_training_forward=True, overlap_moe_expert_parallel_comm=True)

    @pytest.mark.parametrize(
        "override,match",
        [
            ({"moe_router_load_balancing_type": "sinkhorn"}, "would select"),
            ({"moe_input_jitter_eps": 0.01}, "moe_input_jitter_eps"),
            ({"moe_router_force_load_balancing": True}, "force_load_balancing"),
            ({"moe_router_force_biased": 0.5}, "force_load_balancing"),
            # Rejected by the inference-optimized dropless check rather than by a
            # mega-specific one, since this path always implies that impl.
            ({"moe_expert_capacity_factor": 1.0}, "dropless"),
        ],
    )
    def test_rejects_routing_that_breaks_parity(self, override, match):
        """Anything that makes the training router pick different experts.

        Generation routes with plain top-k, so these would silently cost the
        parity the whole feature exists for.
        """
        with pytest.raises(ValueError, match=match):
            _config(moe_mega_training_forward=True, **override)
