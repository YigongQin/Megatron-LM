# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Does the batch-invariant inference activation match the training fusion?

The MoE half of a zero-KL run pairs TE's grouped GEMM in training against the
vLLM fused MoE in generation. Under ``batch_invariant_mode`` that inference
path deliberately *un*-fuses its activation and routes it through
``inference/moe/batch_invariant.py`` instead, whose entire purpose is to
reproduce the training fusion's rounding sequence bit for bit. If it does not,
every routed token is slightly wrong in generation, and the symptom is a small
constant KL that does not grow with training -- which is what the nano-v3 run
shows (0.0025 / 0.0041 / 0.0025 over three steps).

Nothing pinned that claim before this file. ``test_vllm_fused_moe.py`` compares
the *fused* path against a Python reference at ``atol=5e-2``, which is three
orders of magnitude looser than the effect being looked for, and contains no
reference to ``batch_invariant`` at all. So the one kernel that asserts train
parity was the one kernel with no parity test.

These comparisons are bitwise on purpose. A tolerance here would defeat the
point: the question is not whether the two agree to some epsilon but whether
generation reproduces training exactly, because a rollout feeds its own output
back in and a per-token bias does not average away.

SwiGLU is included as a control rather than for its own sake. The qwen zero-KL
arm reaches KL=0 through this same vLLM backend and dispatcher, differing only
in the activation, so if squared ReLU disagrees while SwiGLU agrees, the
activation is confirmed as the cause rather than merely suspected.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernels under comparison are CUDA-only"
)

ROWS = 512
HIDDEN = 768  # nano-v3's moe_ffn_hidden_size
CLAMP_SCALE = 8.0


def _report(label, ours, theirs):
    """Print the comparison even when it passes, and return whether it is bitwise.

    The printed number is the deliverable: a pass tells us the activation is
    eliminated as a KL source, and that is only meaningful if the magnitude is
    visible rather than hidden behind an assertion that did not fire.
    """
    same = torch.equal(ours, theirs)
    diff = (ours.float() - theirs.float()).abs()
    denom = theirs.float().pow(2).mean().sqrt().clamp_min(1e-12)
    rel_rms = (diff.pow(2).mean().sqrt() / denom).item()
    mismatched = int((ours != theirs).sum().item())
    print(
        f"[bi-moe-parity] {label}: bitwise={same} rel_rms={rel_rms:.3e} "
        f"mismatched={mismatched}/{ours.numel()} max_abs={diff.max().item():.3e}"
    )
    return same


def _inputs(width, seed=0):
    """Pre-activations and routing probabilities in their production dtypes.

    FC1 output arrives in BF16, and ``moe_router_dtype: fp32`` in the recipe
    means the probabilities are FP32 -- which matters, because both sides apply
    the probability in FP32 and round once afterwards, so feeding BF16
    probabilities here would test a rounding sequence neither side runs.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # Centred on zero so roughly half the inputs land on the flat side of the
    # ReLU. A positive-only input would exercise only the squaring path and
    # would agree even if the ReLU boundary were handled differently.
    x = torch.randn(ROWS, width, device="cuda", dtype=torch.bfloat16, generator=gen)
    probs = torch.rand(ROWS, device="cuda", dtype=torch.float32, generator=gen)
    return x, probs


def _all_rows_live():
    """The permutation map and row count for "every row is a routed token".

    The inference kernel skips rows whose map entry is negative and leaves them
    untouched in a ``torch.empty_like`` output, so the comparison is only valid
    where it actually wrote. Marking everything live keeps that from becoming a
    comparison against uninitialised memory.
    """
    permutation_map = torch.arange(ROWS, device="cuda", dtype=torch.int32)
    n_used = torch.tensor(ROWS, device="cuda", dtype=torch.int32)
    return permutation_map, n_used


class TestSquaredReluActivationParity:
    """The activation nano-v3 uses, and the prime suspect for its nonzero KL."""

    @pytest.mark.parametrize(
        "clamp_scale",
        [
            pytest.param(None, id="unclamped"),
            pytest.param(CLAMP_SCALE, id="clamped"),
        ],
    )
    def test_inference_kernel_matches_training_fusion(self, clamp_scale):
        """Both branches, because they round in deliberately different places.

        Unclamped, the inference kernel materializes the square in BF16 before
        the probability multiply, on the stated grounds that training's
        ``torch.pow(F.relu(x), 2)`` squares a BF16 tensor and therefore rounds
        there too. Clamped, it stays in FP32 to a single final round. Only one
        of those can be right for a given training path, and which one depends
        on what the ``@jit_fuser`` decorator does to that expression -- if it
        fuses into an FP32 kernel, training does not round at the square and
        the unclamped branch rounds where training does not.
        """
        from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl
        from megatron.core.inference.moe.batch_invariant import squared_relu_with_probs

        x, probs = _inputs(HIDDEN)
        permutation_map, n_used = _all_rows_live()

        inference = squared_relu_with_probs(
            x.clone(), permutation_map, n_used, probs, clamp_scale
        )
        # (ROWS, 1) so it broadcasts across hidden, which is the shape the
        # grouped MLP passes as permuted_probs.
        training = weighted_squared_relu_impl(x.clone(), probs.unsqueeze(-1), clamp_scale)

        label = f"squared relu {'clamped' if clamp_scale else 'unclamped'}: inference vs training"
        assert _report(label, inference, training), (
            "the batch-invariant inference activation does not reproduce the training "
            "fusion bitwise, so every routed token is biased in generation and a zero-KL "
            "run is not possible on a squared-ReLU model as configured. The rounding "
            "sequences to reconcile are the BF16 materialization of the square in "
            "inference/moe/batch_invariant.py against whatever @jit_fuser makes of "
            "torch.pow(F.relu(x), 2) in fusions/fused_weighted_squared_relu.py."
        )


class TestInferenceTrainingForwardConfig:
    """Wiring for ``moe_inference_training_forward``, the backend-agnostic flag.

    Not a parity test. It checks the parts that fail at construction rather than
    in arithmetic -- the flag resolving, the backend allow-list, the mega alias
    still validating as before -- because those are cheap to get wrong and
    expensive to discover, as the nano run's ``bias_activation_fusion`` crash
    showed: ten minutes of a two-node job for a ``__post_init__`` check.

    Deliberately config-only and single-rank. The bitwise question -- does the
    training value pass reproduce generation through the vLLM kernel -- needs EP
    ranks and a generation forward to compare against, and is not answered here.
    """

    @staticmethod
    def _config(**overrides):
        from megatron.core.activations import squared_relu
        from megatron.core.transformer import TransformerConfig

        kwargs = dict(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_ffn_hidden_size=128,
            moe_grouped_gemm=True,
            add_bias_linear=False,
            gated_linear_unit=False,
            activation_func=squared_relu,
            # bias_activation_fusion defaults False in TransformerConfig; the
            # nano crash came from the RL base config setting it True, which is
            # rejected for any activation outside gelu/silu/quick_gelu.
            transformer_impl='inference_optimized',
            # Required by inference_optimized: a BF16 router would cost a
            # per-decode dtype conversion, and its expert module assumes
            # RMSNorm. Both are unrelated to what this class tests, but without
            # them every test here dies before reaching its assertion.
            moe_router_dtype='fp32',
            normalization='RMSNorm',
            inference_grouped_gemm_backend='vllm',
            moe_inference_training_forward=True,
            # The forward stores no activations, so the backward has to come
            # from a recompute pass. Validated, not assumed.
            recompute_granularity='selective',
            recompute_modules=['moe'],
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        kwargs.update(overrides)
        return TransformerConfig(**kwargs)

    def test_vllm_backend_is_accepted(self):
        """The point of the change: a non-mega backend can drive the value pass."""
        config = self._config()
        assert config.moe_inference_training_forward

    def test_mega_flag_resolves_into_the_general_one(self):
        """Back-compat: the working RL arm sets the mega flag by name.

        It has to keep validating exactly as before and has to imply the general
        flag, since every downstream gate now reads that one. If this breaks,
        the mega path silently stops taking its own forward.
        """
        config = self._config(
            inference_grouped_gemm_backend='flashinfer_mega',
            moe_inference_training_forward=False,
            moe_mega_training_forward=True,
            inference_mega_precision='bf16',
            # The rest of this class is squared ReLU, which is the whole reason
            # mega is unavailable to nano. Mega implements only SwiGLU, so the
            # one mega test has to supply it.
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
        )
        assert config.moe_inference_training_forward, (
            "moe_mega_training_forward no longer implies moe_inference_training_forward, "
            "so moe_layer's gate reads False and the mega value pass is skipped -- the "
            "forward would quietly revert to TE and parity would be lost with no error."
        )

    def test_mega_flag_rejects_a_non_mega_backend(self):
        """The mega spelling is specific; the general one is for everything else."""
        with pytest.raises(ValueError, match="mega-specific spelling"):
            self._config(moe_mega_training_forward=True, moe_inference_training_forward=False)

    def test_unsupported_backend_is_rejected_rather_than_silently_wrong(self):
        """'torch' has no training-side weight rebuild.

        Allowing it would read weights the optimizer has since moved, which
        produces a plausible number rather than an error -- the worst failure
        mode for a parity feature.
        """
        with pytest.raises(ValueError, match="rebuild"):
            self._config(inference_grouped_gemm_backend='torch')

    def test_recompute_is_required(self):
        """Without it there is no backward graph for the experts at all."""
        with pytest.raises(ValueError, match="recompute"):
            self._config(recompute_granularity=None, recompute_modules=None)


class TestSwigluActivationParity:
    """The control: the activation that already reaches KL=0 in the qwen arm.

    Not redundant with the squared-ReLU test above. These two tests differ in
    exactly the variable under suspicion, so together they attribute a failure
    rather than just reporting one. If both fail, the problem is the shared
    probability-multiply-and-round tail, not squared ReLU.
    """

    def test_inference_kernel_matches_training_fusion(self):
        """Gate and up halves interleaved as the grouped MLP passes them."""
        from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
        from megatron.core.inference.moe.batch_invariant import swiglu_with_probs

        x, probs = _inputs(2 * HIDDEN)
        permutation_map, n_used = _all_rows_live()

        inference = swiglu_with_probs(x.clone(), permutation_map, n_used, probs)
        training = weighted_bias_swiglu_impl(x.clone(), None, probs.unsqueeze(-1))

        assert _report("swiglu: inference vs training", inference, training), (
            "SwiGLU also fails to match, which moves the problem off squared ReLU and "
            "onto the shared tail both kernels implement: the FP32 probability multiply "
            "and the single BF16 round after it. Note that the qwen zero-KL arm reaches "
            "KL=0 through this same kernel, so a failure here needs reconciling with "
            "that result before it is believed."
        )
