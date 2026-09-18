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
