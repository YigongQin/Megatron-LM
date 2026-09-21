# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Does the Mamba training forward reproduce generation, stage by stage?

Reinforcement learning compares a training log-prob against a generation
log-prob, so any kernel the two paths do not share is bias in the gradient. The
MoE side of that was closed by routing both through one megakernel. This is the
SSM side, and it starts from a different position: the work to make *inference*
self-consistent is already done and well tested -- ``MambaBatchInvariantDecode``
replays buffered decode tokens through the training chunk scan, and
``test_batch_invariant_decode.py`` pins that against a full-sequence scan
bitwise. What has never been measured is whether *training* agrees with either
of them.

Three paths reach the same tokens, and under ``batch_invariant_mode`` they are
meant to be one:

    training       causal_conv1d_fn + mamba_chunk_scan_combined     (pip)
    dyn. prefill   causal_conv1d_varlen_fn + ..._combined_varlen    (in-tree)
    decode         causal_conv1d_update + buffered chunk replay     (pip + in-tree)

Two things make training's agreement non-obvious even after the config is set
correctly. It defaults to the *fused* ``mamba_split_conv1d_scan_combined``,
which no inference path calls, so ``use_mamba_mem_eff_path`` has to be off --
and the unfused path it falls back to is documented in the mixer as neither
used nor functionally tested. And the gate ``z`` enters on opposite sides of
the scan in the two paths unless batch-invariant mode lines them up.

The comparison is per stage -- after the conv, after the scan, after the norm
-- rather than only on the layer output. A single end-to-end number says the
layer disagrees; it does not say which kernel, and with three implementations
per stage that is the whole question. Pairs are reported before they are
asserted, so a failing run leaves the magnitudes in the log.

Run with ``scripts/local/run_mamba_parity_tests.sh``. Needs a GPU and
MAMBA_DETERMINISTIC=1 set before import.
"""

import os

import pytest
import torch

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    from megatron.core.ssm.ops.mamba2.ssd_combined import mamba_chunk_scan_combined_varlen

    HAVE_MAMBA = True
except ImportError:
    HAVE_MAMBA = False

try:
    from causal_conv1d import causal_conv1d_fn

    HAVE_CAUSAL_CONV1D = causal_conv1d_fn is not None
except ImportError:
    HAVE_CAUSAL_CONV1D = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not HAVE_MAMBA, reason="mamba_ssm required"),
]

# Small enough to run in seconds, large enough to cross a chunk boundary --
# which is where the state-passing carry enters and where a scan that agrees
# within one chunk can still diverge.
NHEADS = 8
HEADDIM = 32
NGROUPS = 1
DSTATE = 16
CHUNK_SIZE = 32
SEQLEN = 96  # three chunks
D_CONV = 4


def _report(label, a, b):
    """Log a pair's agreement and return whether it is bitwise identical.

    Reported unconditionally. The number is the finding here even when it
    passes: 'bitwise' and 'agrees to 3e-3' are different answers to whether a
    zero-KL run is possible, and only one of them survives an autoregressive
    rollout.
    """
    same = torch.equal(a, b)
    delta = (a.float() - b.float()).pow(2).mean().sqrt()
    scale = b.float().pow(2).mean().sqrt().clamp_min(1e-12)
    print(f"[mamba-parity] {label}: bitwise={same} rel_rms={(delta / scale).item():.3e}")
    return same


@pytest.fixture(name="inputs")
def _inputs():
    """One set of SSM inputs, shared by every path under test."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(0)

    def randn(*shape, dt=dtype):
        return torch.randn(*shape, device=device, dtype=dt, generator=generator)

    return dict(
        device=device,
        dtype=dtype,
        x=randn(1, SEQLEN, NHEADS, HEADDIM),
        z=randn(1, SEQLEN, NHEADS, HEADDIM),
        dt=randn(1, SEQLEN, NHEADS, dt=torch.float32),
        B=randn(1, SEQLEN, NGROUPS, DSTATE),
        C=randn(1, SEQLEN, NGROUPS, DSTATE),
        A=-torch.rand(NHEADS, device=device, dtype=torch.float32, generator=generator).exp(),
        D=torch.rand(NHEADS, device=device, dtype=torch.float32, generator=generator),
        dt_bias=torch.rand(NHEADS, device=device, dtype=torch.float32, generator=generator),
    )


class TestDeterminismPrecondition:
    """The environment has to be right before any comparison means anything."""

    def test_mamba_deterministic_is_set(self):
        """Fail loudly rather than measure a kernel that retunes per shape.

        ``autotune_configs`` runs inside the ``@triton.autotune`` decorator, so
        the config list is fixed when the module is imported. A run that starts
        without this set cannot be fixed afterwards, and the symptom would be
        an intermittent bitwise failure that looks like a kernel bug.
        """
        from megatron.core.ssm.ops.common.determinism import use_deterministic_mode

        assert use_deterministic_mode(), (
            "set MAMBA_DETERMINISTIC=1 before running; the SSM Triton kernels pin "
            f"their autotune configs at import (MAMBA_DETERMINISTIC="
            f"{os.environ.get('MAMBA_DETERMINISTIC')!r})"
        )


class TestScanParity:
    """Training's chunk scan against the varlen scan dynamic prefill runs.

    Training and decode are expected to agree already: the unfused training
    path calls ``mamba_chunk_scan_combined``, and that same function is the
    reference the batch-invariant decode tests pin against. Dynamic prefill is
    the one that has never been compared -- ``test_ssd_combined.py`` checks its
    shapes and that it produces no NaN, and nothing else.

    Prefill is also the path where a difference matters most, because it
    produces the state the first generated token is decoded from. An error
    there does not stay local; it seeds every token that follows.
    """

    def test_training_scan_matches_varlen_prefill(self, inputs):
        """The finding this whole file exists to produce."""
        x, z, dt, B, C = (inputs[k] for k in ("x", "z", "dt", "B", "C"))
        A, D, dt_bias = inputs["A"], inputs["D"], inputs["dt_bias"]

        # Training, unfused, batch-invariant convention: z through the scan.
        y_train, _ = mamba_chunk_scan_combined(
            x, dt, A, B, C, CHUNK_SIZE, D=D, z=z, dt_bias=dt_bias,
            dt_softplus=True, return_final_states=True,
        )

        # Dynamic prefill: one request spanning the whole sequence, chunked the
        # same way, so the only difference from the call above is the kernel.
        # The varlen entry point writes into a preallocated output and returns
        # the boundary states, rather than returning the sequence.
        device = inputs["device"]
        num_chunks = SEQLEN // CHUNK_SIZE
        cu_chunk_seqlens = torch.arange(
            0, SEQLEN + 1, CHUNK_SIZE, dtype=torch.int32, device=device
        )
        last_chunk_indices = torch.tensor([num_chunks - 1], dtype=torch.int64, device=device)
        seq_idx = torch.zeros(num_chunks, dtype=torch.int32, device=device)
        y_prefill = torch.empty_like(x.squeeze(0))
        mamba_chunk_scan_combined_varlen(
            x=x.squeeze(0),
            dt=dt.squeeze(0),
            A=A,
            B=B.squeeze(0),
            C=C.squeeze(0),
            chunk_size=CHUNK_SIZE,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=y_prefill,
            D=D,
            z=z.squeeze(0),
            dt_bias=dt_bias,
            dt_softplus=True,
        )

        same = _report(
            "training chunk scan vs varlen prefill scan",
            y_train.squeeze(0).reshape(-1),
            y_prefill.reshape(-1),
        )
        assert same, (
            "dynamic prefill does not reproduce the training scan bitwise, so a "
            "zero-KL run on this model is not possible as configured. The prefill "
            "state seeds every decoded token, so this is not a bounded error. Fix "
            "by pinning mamba_chunk_scan_combined_varlen against pip, or by routing "
            "prefill through the dense kernel."
        )


@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal-conv1d required")
class TestConvParity:
    """The depthwise conv, which has three implementations across the paths.

    Its existing test compares the varlen fork to pip at ``atol=1e-2`` for
    bf16 -- a tolerance, which is the right bar for a kernel port and the wrong
    one for train/generation parity. This asks the stricter question.
    """

    def test_training_conv_matches_varlen_prefill_conv(self, inputs):
        """Bitwise, not close."""
        from megatron.core.ssm.ops.common.causal_conv1d_varlen import causal_conv1d_varlen_fn

        device, dtype = inputs["device"], inputs["dtype"]
        conv_dim = NHEADS * HEADDIM
        generator = torch.Generator(device=device).manual_seed(1)
        xBC = torch.randn(SEQLEN, conv_dim, device=device, dtype=dtype, generator=generator)
        weight = torch.randn(conv_dim, D_CONV, device=device, dtype=dtype, generator=generator)
        bias = torch.randn(conv_dim, device=device, dtype=dtype, generator=generator)
        cu_seqlens = torch.tensor([0, SEQLEN], dtype=torch.int32, device=device)

        y_train = causal_conv1d_fn(
            x=xBC.t().unsqueeze(0).contiguous(), weight=weight, bias=bias, activation="silu"
        )
        y_prefill = causal_conv1d_varlen_fn(
            x=xBC, weight=weight, bias=bias, cu_seqlens=cu_seqlens, activation="silu"
        )

        same = _report(
            "training conv vs varlen prefill conv",
            y_train.squeeze(0).t().reshape(-1),
            y_prefill.reshape(-1),
        )
        assert same, (
            "the varlen conv fork is not bitwise-equal to the pip conv the training "
            "forward runs; its own test only requires atol=1e-2, which is not enough "
            "for train/generation parity"
        )


class TestMixerTrainingMatchesPrefill:
    """The same question as TestScanParity, one level up: does the *layer* agree?

    Everything above compares kernels called directly on synthetic tensors. That
    leaves the wiring untested -- which projection feeds which kernel, where the
    gate enters, whether the norm sees it, how the state is threaded -- and the
    wiring is where the equivalent MoE bug actually lived. Every kernel-level
    mega test passed while generation and training disagreed, because the
    adapter handed experts to the kernel in score order on one path and index
    order on the other. Same kernel, same inputs, different arithmetic.

    So this drives a real ``MambaMixer``: a training forward against the
    dynamic-inference prefill, through ``in_proj`` and the gated norm, on one
    set of weights. It is also the only coverage of the ``batch_invariant_mode``
    branch in ``_static_prefill``, which is the one line of behaviour this work
    changed.

    Decode is deliberately not repeated here. It reduces to
    ``mamba_chunk_scan_combined`` by construction and
    ``test_batch_invariant_decode.py`` already pins that bitwise; the untested
    edge was always training versus prefill.
    """

    @staticmethod
    def _build_mixer():
        """A real mixer configured the way a parity run configures one."""
        import torch.nn.functional as F

        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.ssm.mamba_mixer import MambaMixer
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.enums import AttnBackend
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            activation_func=F.silu,
            # The parity configuration, not a default one. is_hybrid_model is
            # what makes __post_init__ apply the SSM-specific checks, so
            # building this config is itself a test of them.
            is_hybrid_model=True,
            use_mamba_mem_eff_path=False,
            batch_invariant_mode=True,
            batch_invariant_backend="te_native",
            attention_backend=AttnBackend.flash,
            flash_attention_version=4,
            # Required: batch-invariant mode rejects attention dropout, and
            # TransformerConfig defaults it to 0.1. hidden_dropout goes with it
            # for a second reason -- the training side of this comparison runs
            # a module in train mode, and any live dropout would make the two
            # forwards differ for a reason that has nothing to do with parity.
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        submodules = hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            config,
            submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
        )
        return mixer.cuda().to(torch.bfloat16)

    @staticmethod
    def _prefill_context(seqlen, device):
        """The varlen metadata for one request spanning the whole sequence.

        Chunk metadata is left unset on purpose: that selects the fallback
        which rebuilds chunk boundaries from cu_seqlens, so this test does not
        depend on the scheduler agreeing with it. No slot allocator means
        intermediate extraction (prefix caching) stays off, which is also how
        the parity recipe runs it.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            mamba_metadata=SimpleNamespace(
                seq_idx=torch.zeros((1, seqlen), dtype=torch.int32, device=device),
                cu_seqlens=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
                # int32, matching how the real context allocates it. The BIK
                # seed path asserts on the dtype rather than casting.
                batch_indices_prefill=torch.tensor([0], dtype=torch.int32, device=device),
                intermediate_chunk_indices=None,
                intermediate_abs_positions=None,
                intermediate_real_count=None,
                cu_chunk_seqlens=None,
                last_chunk_indices=None,
                seq_idx_for_varlen=None,
                conv_seq_idx=None,
                conv_seq_start=None,
            ),
            mamba_slot_allocator=None,
        )

    def _compare(self, mixer, ssm_state_dtype, label):
        """Training forward vs dynamic prefill, with the state cache dtype varied.

        ``ssm_prefill`` forwards ``ssm_state.dtype`` to the scan as
        ``state_dtype``, so this argument decides the precision of the
        inter-chunk carry on the prefill side. Training's carry is whatever the
        installed mamba_ssm materializes and is not selectable here, which is
        the whole reason this is a parameter: matching it isolates the wiring,
        and not matching it reproduces what a real run does.
        """
        device = torch.device("cuda")
        # Two chunks, so the state-passing carry between them is exercised. A
        # sequence inside a single chunk can agree while the carry does not,
        # and the carry is what a rollout depends on.
        seqlen = 2 * mixer.chunk_size
        torch.manual_seed(0)
        hidden = torch.randn(
            seqlen, 1, mixer.config.hidden_size, device=device, dtype=torch.bfloat16
        )

        train_out, _ = mixer(hidden)

        # Mirror forward()'s tail rather than calling forward again: the
        # inference branch needs a context, and reproducing the same three
        # steps (project, run the SSM, project back) is what makes the two
        # sides comparable at the layer's output.
        conv_shape, ssm_shape = mixer.mamba_state_shapes_per_request()
        conv_state = torch.zeros(1, *conv_shape, device=device, dtype=torch.bfloat16)
        ssm_state = torch.zeros(1, *ssm_shape, device=device, dtype=ssm_state_dtype)
        with torch.inference_mode():
            zxBCdt, _ = mixer.in_proj(hidden)
            y = mixer.ssm_prefill(
                zxBCdt=zxBCdt,
                conv_state=conv_state,
                ssm_state=ssm_state,
                context=self._prefill_context(seqlen, device),
            )
            prefill_out, _ = mixer.out_proj(y.reshape(seqlen, 1, -1))

        return _report(label, train_out.reshape(-1), prefill_out.reshape(-1))

    def test_training_forward_matches_dynamic_prefill(self):
        """With the carry precision matched, does the wiring agree bitwise?

        BF16 cache, so prefill's ``state_dtype`` matches what the pinned
        mamba_ssm materializes for training. That removes the one known
        precision asymmetry and leaves only the wiring under test, which is
        what this class is for.
        """
        mixer = self._build_mixer()
        try:
            same = self._compare(
                mixer, torch.bfloat16, "mixer training vs prefill (carry matched, bf16)"
            )
            assert same, (
                "the mixer's training forward does not reproduce its own prefill "
                "bitwise even with the carry precision matched, so the cause is the "
                "wiring rather than dtype: gate placement in _static_prefill, the "
                "gated norm, or the conv state. This is the MoE expert-ordering "
                "failure in a different layer, and comparing kernels cannot find it."
            )
        finally:
            from tests.unit_tests.test_utilities import Utils

            Utils.destroy_model_parallel()

    def test_fp32_state_cache_does_not_change_the_output(self):
        """The dtype production actually uses, which is the one that matters.

        ``batch_invariant_mode`` forces the SSM state cache to FP32 so decode
        resumes from an unrounded boundary, and ``ssm_prefill`` passes that
        dtype down as the scan's ``state_dtype``. Training cannot match it --
        the pinned mamba_ssm hardcodes ``out_dtype=C.dtype`` -- so the two sides
        do store the inter-chunk states at different precision.

        It does not follow that their outputs differ, and they do not. The scan
        consumes a BF16 view of those states on the output path either way, so
        the dtype governs the snapshot decode resumes from, not the activations
        this comparison sees. The same reasoning is spelled out for the GDP
        kernel in ssm/ops/gdp/chunk.py.

        Worth pinning rather than assuming: it is the reason a hybrid parity run
        needs no mamba_ssm upgrade, and if it ever stops holding, output parity
        silently acquires a dependency on the state cache dtype.
        """
        mixer = self._build_mixer()
        try:
            same = self._compare(
                mixer, torch.float32, "mixer training vs prefill (fp32 cache, as production)"
            )
            assert same, (
                "prefill output now depends on the SSM state cache dtype, which it "
                "did not before: at FP32 it no longer matches the training forward, "
                "while the BF16 comparison above still passes. The scan has started "
                "consuming the states at their stored precision on the output path, "
                "so training must now match it -- which needs a mamba_ssm exposing "
                "state_dtype (state-spaces/mamba#972, the rev Megatron-LM pins) plus "
                "mamba_training_ssm_states_dtype=float32."
            )
        finally:
            from tests.unit_tests.test_utilities import Utils

            Utils.destroy_model_parallel()


class TestGatePlacement:
    """The two places the gate can enter, and what choosing wrongly costs.

    Not a bug hunt: this quantifies a mismatch we know exists by construction,
    so that the config assertion forbidding it has a number attached rather
    than an argument. Training without batch-invariant mode gates after the
    scan; every inference path gates inside it.
    """

    def test_gate_inside_and_outside_the_scan_differ(self, inputs):
        """If these were equal, the mixer change would be unnecessary."""
        x, z, dt, B, C = (inputs[k] for k in ("x", "z", "dt", "B", "C"))
        A, D, dt_bias = inputs["A"], inputs["D"], inputs["dt_bias"]
        common = dict(D=D, dt_bias=dt_bias, dt_softplus=True, return_final_states=True)

        y_inside, _ = mamba_chunk_scan_combined(x, dt, A, B, C, CHUNK_SIZE, z=z, **common)
        y_outside, _ = mamba_chunk_scan_combined(x, dt, A, B, C, CHUNK_SIZE, z=None, **common)
        # The mixer's outside-gating applies RMSNormGated rather than a bare
        # multiply, so this is a lower bound on the divergence, not a model of
        # it. A lower bound is all the claim needs.
        y_outside = y_outside * torch.nn.functional.silu(z)

        _report("gate inside the scan vs gate applied after it", y_inside, y_outside)
        assert not torch.equal(y_inside, y_outside), (
            "gating inside and outside the scan produced identical bits, so the "
            "batch_invariant_mode branch added to _static_prefill is unnecessary "
            "and the config assertion that depends on it should be revisited"
        )
