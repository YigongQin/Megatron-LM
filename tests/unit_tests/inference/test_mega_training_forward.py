# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed tests for the mega MoE training forward (moe_mega_training_forward).

Three properties matter for the parity use case, and they are what these tests
cover:

1. The path runs at all under the parallelism it is meant for: expert parallel
   crossed with data parallel, which is where the shared kernel-weight scratch
   and the per-pass dispatcher swap actually get exercised.
2. The training forward reproduces the generation forward. This is the whole
   point: in RL the importance ratio compares a training log-prob against a
   generation log-prob, so any kernel difference between the two shows up as
   bias that no amount of tuning removes.
3. Forward and backward stay close to the standard TE bf16 path, so the
   gradient taken through the recompute pass is still the gradient of
   something close to what was computed.
4. The MoE layer behaves the same inside a whole transformer layer as it does
   alone. Everything else here builds the MoE layer by itself, so nothing sees
   attention, the projections, or the norms around it -- and enabling the
   megakernel on the training side also switches that surrounding layer to the
   inference-optimized spec, which is a change the MoE-only tests cannot
   observe. Generation is out of scope at this level: it needs a dynamic
   inference context and a KV cache, which only the end-to-end run provides.
5. Generation keeps serving the current weights. RL refits between rollouts and
   the megakernel holds its own transformed copy of the expert weights, so a
   refit that does not reach that copy means the next rollout samples from the
   previous step's policy -- silently, since nothing reads the parameters again.
   This is the one property here that is about generation alone, and it is
   covered in this file because it shares the harness.

Needs Blackwell for the sm100 megakernels, plus the FlashInfer main-branch
overlay that carries moe_ep. ``scripts/local/run_mega_training_tests.sh`` sets
that environment up; running pytest directly against a stock container will
skip everything. MEGA_TEST_EP_SIZE overrides the expert-parallel size, which
defaults to 4.

Data parallel is out of scope here, so EP spans the whole world and
test_expert_data_parallel_replicas_agree skips itself. FlashInfer's NVSHMEM
bootstrap picks the group's rank 0 to mint the unique id but broadcasts it with
``dist.broadcast(src=0, group=ep_group)``, where ``src`` is a global rank, so
only the EP group containing global rank 0 can bootstrap and any other group
hangs the run. Megatron owns that handshake rather than FlashInfer: bringing
NVSHMEM up over the EP group first -- translating the root with
``dist.get_global_rank``, as ``resharding.copy_services.nccl_m2n`` already does
-- makes FlashInfer's ``_ensure_nvshmem`` early-return and skip its own
bootstrap entirely.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.moe.mega._deps import _HAVE_FLASHINFER_MOE_EP
from megatron.core.inference.moe.mega.training_weights import MegaTrainingWeightScratch
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.token_dispatcher_inference import (
    MegaLocalPassthroughDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Expert parallel size. Keep this equal to the world size; see the module
# docstring for why data-parallel replication of expert shards hangs.
EP_SIZE = int(os.environ.get("MEGA_TEST_EP_SIZE", "4"))
LOCAL_TOKENS = 8
# topk=8 is DS-V3's value, and it is the default here because the two paths
# obtain their expert indices differently: generation reads them from
# torch.topk on the router scores, so probability-descending, while training
# recovers them from TopKRouter's boolean map. At topk=2 the combine is a
# two-term sum, which commutes exactly and cannot see a difference in that
# order; at topk=8 it can. Parity is bitwise either way, so the kernel's output
# does not depend on the order, but the geometry that could expose it is the
# one worth testing. MEGA_TEST_TOPK / MEGA_TEST_NUM_EXPERTS override.
NUM_EXPERTS = int(os.environ.get("MEGA_TEST_NUM_EXPERTS", "16"))
ROUTER_TOPK = int(os.environ.get("MEGA_TEST_TOPK", "8"))
# Token counts for the batch-invariance check. Generation decodes a few tokens
# per rank while a training forward takes a whole microbatch, so RL parity rests
# on a token's output not depending on how many others shared the launch. Every
# other test here uses one count on both sides and so cannot see that.
GEN_TOKENS = int(os.environ.get("MEGA_TEST_GEN_TOKENS", "8"))
TRAIN_TOKEN_COUNTS = tuple(
    int(count) for count in os.environ.get("MEGA_TEST_TRAIN_TOKENS", "512,2048").split(",")
)
# Batch-invariant mode composes with the megakernel: mega replaces only the
# expert compute while batch-invariant mode covers attention, the projections
# and the router. The router is the part that matters here -- it is the one
# place the two paths still run different code, since InferenceTopKRouter uses
# a torch.compile'd top-k unless batch-invariant mode swaps in the same eager
# function the training TopKRouter calls. Off by default so the existing
# measurements stay comparable; MEGA_TEST_BATCH_INVARIANT=1 turns it on.
BATCH_INVARIANT = os.environ.get("MEGA_TEST_BATCH_INVARIANT", "0") == "1"
BATCH_INVARIANT_BACKEND = os.environ.get("MEGA_TEST_BI_BACKEND", "te_native")
# How many times the train/gen parity forward is repeated. The forwards are tiny,
# so this is cheap next to layer construction and kernel compilation.
PARITY_REPEATS = int(os.environ.get("MEGA_TEST_PARITY_REPEATS", "20"))
# How many fresh layer pairs the construction-stability test builds. Repeated
# forwards on one pair agree bitwise while separately built pairs have been seen
# to disagree, so this is the axis the variance actually lives on.
PARITY_BUILDS = int(os.environ.get("MEGA_TEST_PARITY_BUILDS", "5"))


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not _HAVE_FLASHINFER_MOE_EP, reason="FlashInfer moe_ep mega not installed"),
    pytest.mark.skipif(not _is_blackwell(), reason="sm100 mega kernels require Blackwell"),
    pytest.mark.skipif(
        torch.cuda.device_count() < EP_SIZE, reason=f"EP={EP_SIZE} needs {EP_SIZE} GPUs"
    ),
    pytest.mark.skipif(
        NUM_EXPERTS % EP_SIZE != 0 or ROUTER_TOPK > NUM_EXPERTS,
        reason=f"num_experts={NUM_EXPERTS} must divide EP={EP_SIZE} and be >= {ROUTER_TOPK}",
    ),
]


def _config(mega_training: bool, **overrides):
    """Config for the inference-optimized mega MoE layer.

    ``mega_training`` selects the parity path; everything else is held fixed so
    the two only differ in which kernel produces the forward value.
    """
    base = dict(
        num_layers=1,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=128,
        moe_router_topk=ROUTER_TOPK,
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
        # Batch-invariant mode requires FlashAttention pinned to v3/v4 and no
        # attention dropout. Nothing here builds attention -- the spec is
        # MoE-only -- but the config validates those fields regardless.
        attention_backend=AttnBackend.flash if BATCH_INVARIANT else AttnBackend.local,
        flash_attention_version=3 if BATCH_INVARIANT else None,
        attention_dropout=0.0,
        batch_invariant_mode=BATCH_INVARIANT,
        batch_invariant_backend=BATCH_INVARIANT_BACKEND,
        expert_model_parallel_size=EP_SIZE,
        recompute_granularity="selective",
        recompute_modules=["moe"],
        moe_mega_training_forward=mega_training,
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _build_layer(config, for_inference: bool = False):
    """Build the MoE layer.

    ``for_inference`` allocates the valid-tokens scalar that the dynamic
    inference context normally owns. Training-only layers deliberately leave it
    unallocated, which is the real situation the parity path has to tolerate.
    """
    if for_inference:
        MegaLocalPassthroughDispatcher.allocate_buffers()
    return get_inference_optimized_moe_spec()(config=config).cuda()


def _copy_expert_weights(src, dst):
    """Give two layers identical expert and router weights.

    Done before either layer's first forward, while parameters still own their
    storage: the inference path redirects ``param.data`` into a concatenated
    buffer on first use.
    """
    dst.load_state_dict(src.state_dict())


def _bump_expert_weights(layer, delta=0.01):
    """Rewrite the expert parameters in place, standing in for an optimizer or a refit.

    In place on purpose. Once generation has run, the parameters are views into
    the concatenated stack that the kernel weights are packed from, so this is
    the same write a refit performs and it reaches the buffer the repack reads.
    """
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "expert" in name or "linear_fc" in name:
                param.add_(delta)


def _use_flashinfer_preprocessing(layer):
    """Put one generation layer back on FlashInfer's own weight preprocessing.

    bf16 generation owns its transformed weights so that a refit can rewrite
    them, which leaves no in-tree way to reach the path FlashInfer takes when it
    preprocesses and snapshots the weights itself. Rebuilding the adapter is
    enough to get it back: each one builds its FlashInfer layer lazily, so this
    is still a layer that has never been handed a weight.
    """
    from megatron.core.inference.moe.mega import MegatronMegaMoEAdapter

    experts = layer.experts
    experts._mega_adapter = MegatronMegaMoEAdapter(
        config=experts.config, ep_group=experts.ep_group, owns_transformed_weights=False
    )
    # Instance attribute shadowing the method, so the adapter is handed the raw
    # per-expert stacks and preprocesses them the way it did before the buffer.
    experts._mega_inference_weights = lambda: None


def _hidden(config, seed, tokens=None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        LOCAL_TOKENS if tokens is None else tokens,
        1,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )


def _chunked_generation(layer, hidden, chunk):
    """Replay ``hidden`` through the generation layer ``chunk`` tokens at a time."""
    with torch.no_grad(), InferenceMode.active():
        return torch.cat(
            [
                layer(hidden[start : start + chunk])[0]
                for start in range(0, hidden.shape[0], chunk)
            ],
            dim=0,
        )


def _rel_rms(got, want):
    """Relative RMS error, which is the right scale-free metric for a whole tensor.

    Max-abs relative error is dominated by near-zero entries and is not
    informative for a reduction as wide as an MoE layer output.
    """
    got, want = got.float(), want.float()
    denom = want.pow(2).mean().sqrt()
    if denom == 0:
        return (got - want).pow(2).mean().sqrt().item()
    return ((got - want).pow(2).mean().sqrt() / denom).item()


def _measure(got, want, label):
    """Worst relative RMS error across ranks, reported so a pass shows its margin.

    Each rank holds different tokens and different local experts, so the honest
    number is the worst one rather than whichever rank pytest happens to quote.
    Asserting on the reduced value also makes the verdict rank-independent.

    Collective: every rank must reach this, which holds because these tests take
    the same path on all ranks.
    """
    error = torch.tensor([_rel_rms(got, want)], device="cuda")
    torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.MAX)
    error = error.item()
    if torch.distributed.get_rank() == 0:
        print(f"[mega-metric] {label}: rel_rms={error:.3e}", flush=True)
    return error


def _routing_diff(gen_layer, train_layer, hidden, num_experts):
    """Compare the routing the two paths hand the kernel, token by token.

    The weight bytes are already known to agree, since parity at a single token
    count is bitwise, which leaves routing as the remaining asymmetry:
    generation's router returns topk indices directly while training recovers
    them from TopKRouter's boolean map, and the two reach their probabilities
    through different reductions.

    Returns how many tokens pick a different expert set, how many list the same
    set in a different order, and the largest per-expert probability gap.
    """
    with torch.no_grad(), InferenceMode.active():
        gen_probs, gen_ids = gen_layer.router(hidden)
    with torch.no_grad():
        dense_probs, dense_map = train_layer.router(hidden)
        train_ids, train_probs = train_layer.experts._mega_dense_routing_to_topk(
            dense_map, dense_probs
        )

    def by_expert(ids, probs):
        """Scatter [tokens, topk] onto [tokens, num_experts] so ids align."""
        out = torch.zeros(
            ids.shape[0], num_experts, device=ids.device, dtype=torch.float32
        )
        return out.scatter_(1, ids.long(), probs.float())

    gen_dense, train_dense = by_expert(gen_ids, gen_probs), by_expert(train_ids, train_probs)
    set_differs = int(
        ((gen_dense != 0) != (train_dense != 0)).any(dim=1).sum().item()
    )
    order_differs = int((gen_ids != train_ids).any(dim=-1).sum().item())
    return set_differs, order_differs, (gen_dense - train_dense).abs().max().item()


def _measure_token_count_parity(batched, chunked, label):
    """Compare one wide forward against the same tokens taken a few at a time.

    Reports the per-token count as well as the aggregate: an rms over thousands
    of tokens dilutes a handful of badly wrong ones, and how many tokens moved
    is what separates a reduction-order wobble (most tokens off by a little)
    from a packing or tile-boundary bug (a few off by a lot).
    """
    count = batched.shape[0]
    differed = int((batched != chunked).flatten(1).any(dim=1).sum().item())
    stats = torch.tensor([_rel_rms(batched, chunked), float(differed)], device="cuda")
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
    error, worst_differed = stats[0].item(), int(stats[1].item())
    if torch.distributed.get_rank() == 0:
        print(
            f"[mega-metric] token-count parity ({label}): batched={count} "
            f"chunk={GEN_TOKENS} rel_rms={error:.3e} "
            f"({worst_differed}/{count} tokens differ)",
            flush=True,
        )
    return error, worst_differed



def _measure_scalar(value, label):
    """Reduce and report one already-computed number, the way _measure does tensors."""
    stats = torch.tensor([value], device="cuda", dtype=torch.float64)
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
    worst = stats[0].item()
    if torch.distributed.get_rank() == 0:
        print(f"[mega-metric] {label}: rel_rms={worst:.3e}", flush=True)
    return worst


@pytest.fixture(autouse=True)
def _parallel_state():
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
    )

    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=EP_SIZE)
    # Global kernel patching, so it must be on before any layer is built and off
    # again afterwards -- leaving it on would silently change the next test.
    if BATCH_INVARIANT:
        enable_batch_invariant_mode(backend=BATCH_INVARIANT_BACKEND)
    # Logged as a metric so a log says which geometry produced its numbers;
    # sort -u in the runner collapses the repeats across tests.
    if torch.distributed.get_rank() == 0:
        print(
            f"[mega-metric] geometry: experts={NUM_EXPERTS} topk={ROUTER_TOPK} "
            f"ep={EP_SIZE} local_tokens={LOCAL_TOKENS} "
            f"batch_invariant={BATCH_INVARIANT}"
            + (f" ({BATCH_INVARIANT_BACKEND})" if BATCH_INVARIANT else ""),
            flush=True,
        )
    # Registers the 'expert-parallel-rng' tracker state that expert weight init
    # requires. Utils.initialize_model_parallel does not do this.
    model_parallel_cuda_manual_seed(123)
    MegaTrainingWeightScratch.reset()
    # Both are process-wide; leaking them would let one test's inference setup
    # mask a missing allocation in the next.
    InferenceAllGatherDispatcherBase._valid_tokens_tensor = None
    yield
    MegaTrainingWeightScratch.reset()
    InferenceAllGatherDispatcherBase._valid_tokens_tensor = None
    if BATCH_INVARIANT:
        disable_batch_invariant_mode()
    Utils.destroy_model_parallel()


class TestMegaTrainingForwardRuns:
    """(1) The path runs under expert parallel crossed with data parallel."""

    def test_forward_backward_runs_under_ep_and_dp(self):
        config = _config(mega_training=True)
        layer = _build_layer(config).train()
        hidden = _hidden(config, seed=0).requires_grad_(True)

        out, _ = layer(hidden)
        out.sum().backward()

        assert out.shape == hidden.shape
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
        # The backward must reach the expert weights through the recompute pass,
        # not just the input.
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        fc1_grad = layer.experts.linear_fc1.weight0.grad
        assert fc1_grad is not None, "expert weights received no gradient"
        assert torch.isfinite(fc1_grad).all()
        assert fc1_grad.abs().sum() > 0

    def test_expert_data_parallel_replicas_agree(self):
        """Ranks holding the same experts must produce the same output.

        The expert-data-parallel group is the set of ranks that replicate the
        same expert shard, so with identical weights and identical input they
        must agree. A disagreement means the shared scratch or the dispatcher
        swap is rank-dependent.
        """
        from megatron.core import parallel_state

        config = _config(mega_training=True)
        layer = _build_layer(config).train()

        edp_group = parallel_state.get_expert_data_parallel_group()
        if torch.distributed.get_world_size(edp_group) == 1:
            pytest.skip("no expert-data-parallel replication; run with 2 x EP_SIZE ranks")
        # Sync weights within the replica group only, so distinct EP ranks keep
        # distinct expert shards and the EP split stays realistic.
        src = torch.distributed.get_global_rank(edp_group, 0)
        for param in layer.parameters():
            torch.distributed.broadcast(param.data, src=src, group=edp_group)

        out, _ = layer(_hidden(config, seed=0))

        gathered = [
            torch.empty_like(out) for _ in range(torch.distributed.get_world_size(edp_group))
        ]
        torch.distributed.all_gather(gathered, out.contiguous(), group=edp_group)
        for other in gathered[1:]:
            assert torch.equal(gathered[0], other)


class TestTrainGenParity:
    """(2) The training forward reproduces the generation forward."""

    # The cold forward is held looser than the warm one only because it has been
    # seen to land at ~8e-4 in a minority of process launches while the warm
    # forward stayed bitwise. That residual is unexplained and reproduces without
    # any of this code (two separately built generation layers have shown it), so
    # it is bounded rather than asserted away. 5e-3 is about one bf16 ulp, well
    # under the O(0.1) that stale kernel weights, a wrong expert, or divergent
    # routing would produce.
    COLD_TOL = 5e-3

    def test_training_forward_matches_generation_forward(self):
        config = _config(mega_training=True)
        gen_layer = _build_layer(_config(mega_training=False), for_inference=True).eval()
        train_layer = _build_layer(config).train()
        _copy_expert_weights(gen_layer, train_layer)

        hidden = _hidden(config, seed=1)

        def forward_both():
            with torch.no_grad(), InferenceMode.active():
                gen_out, _ = gen_layer(hidden)
            train_out, _ = train_layer(hidden)
            return train_out, gen_out

        # Repeated because parity has to hold on every step, not on average: a
        # single run of zeros cannot tell bitwise from usually-bitwise, and one
        # run did measure ~8e-4 here with no code change. The first iteration is
        # also the forward that runs the adapter's warmup(), so it is reported
        # apart from the steady state.
        errors = []
        for _ in range(PARITY_REPEATS):
            errors.append(_rel_rms(*forward_both()))

        # One collective rather than one per repeat, so the ranks stay in step.
        # Max across ranks for the same reason as _measure: each rank holds
        # different tokens and experts, so the worst rank is the honest number.
        stats = torch.tensor(
            [errors[0], max(errors[1:], default=0.0), float(sum(e != 0.0 for e in errors))],
            device="cuda",
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
        cold, warm, differed = stats[0].item(), stats[1].item(), int(stats[2].item())
        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] train/gen parity: cold={cold:.3e} worst_warm={warm:.3e} "
                f"({differed}/{PARITY_REPEATS} repeats differed at all)",
                flush=True,
            )

        # Same kernel, same weights, same token count, and a repack meant to
        # reproduce FlashInfer's own preprocessing exactly, so the steady state
        # should be bitwise. Held as a tolerance rather than torch.equal only so
        # a regression reports a number instead of a bare failure; 1e-6 is far
        # below anything the two paths have been observed to differ by.
        assert warm < 1e-6, f"train/gen parity broken when warm: rel_rms={warm:.3e}"
        assert cold < self.COLD_TOL, f"cold train/gen parity: rel_rms={cold:.3e}"

    def test_eval_mode_forward_matches_generation_forward(self):
        """The log-prob pass runs under eval(), and it is the pass RL compares.

        Distinct from the test above, which exercises train(). RL takes
        log-probs with model.eval() and no grad, so a mega forward conditioned
        on training mode would send the one forward whose numbers must match
        generation down the TE path instead, ~6e-3 away, while every test here
        that calls .train() kept passing.
        """
        config = _config(mega_training=True)
        gen_layer = _build_layer(_config(mega_training=False), for_inference=True).eval()
        # eval(), exactly as the log-prob pass leaves it. No recompute pairs
        # with this forward and none is needed: without grad there is no
        # backward to rebuild.
        eval_layer = _build_layer(config).eval()
        _copy_expert_weights(gen_layer, eval_layer)

        hidden = _hidden(config, seed=1)

        errors = []
        for _ in range(PARITY_REPEATS):
            with torch.no_grad():
                with InferenceMode.active():
                    gen_out, _ = gen_layer(hidden)
                eval_out, _ = eval_layer(hidden)
            errors.append(_rel_rms(eval_out, gen_out))

        stats = torch.tensor(
            [errors[0], max(errors[1:], default=0.0)], device="cuda"
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
        cold, warm = stats[0].item(), stats[1].item()
        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] eval/gen parity: cold={cold:.3e} worst_warm={warm:.3e}",
                flush=True,
            )

        assert warm < 1e-6, (
            f"eval-mode (log-prob) forward does not match generation: rel_rms={warm:.3e}. "
            "A value near 6e-3 means the eval forward fell back to TE."
        )
        assert cold < self.COLD_TOL, f"cold eval/gen parity: rel_rms={cold:.3e}"

    def test_parity_holds_across_token_counts(self):
        """The parity RL actually needs: same weights, different token counts.

        Generation decodes a handful of tokens per rank; the training forward
        that recomputes those log-probs sees the whole microbatch. The
        importance ratio compares the two, so a token's output has to be
        independent of how many others rode along. That is not free: the kernel
        packs tokens into its dispatch pool and schedules cluster tiles by token
        count, either of which can change the reduction order.

        Generation replays the same tokens in chunks of GEN_TOKENS, which is
        both how the engine would produce them and the shape of the fallback if
        this fails -- an MoE layer is token-independent given routing, so
        chunking the training forward to the generation count would restore
        parity by construction.

        One axis is deliberately held fixed: both layers get the same workspace
        capacity, so this measures the actual token count rather than the
        compile-time buffer size. Capacity is the second axis and is not covered
        here, since a real deployment could differ on that too.
        """
        # Sized for the largest count so a single capacity serves every
        # comparison; the adapter rejects a forward wider than its workspace.
        capacity = max(GEN_TOKENS, *TRAIN_TOKEN_COUNTS)
        gen_config = _config(
            mega_training=False, inference_mega_max_tokens_per_rank=capacity
        )
        train_config = _config(
            mega_training=True, inference_mega_max_tokens_per_rank=capacity
        )
        gen_layer = _build_layer(gen_config, for_inference=True).eval()
        train_layer = _build_layer(train_config).train()
        _copy_expert_weights(gen_layer, train_layer)

        results = []
        for count in TRAIN_TOKEN_COUNTS:
            hidden = _hidden(train_config, seed=5, tokens=count)
            train_out, _ = train_layer(hidden)
            gen_out = _chunked_generation(gen_layer, hidden, GEN_TOKENS)
            error, differed = _measure_token_count_parity(train_out, gen_out, "train vs gen")
            sets, orders, gap = _routing_diff(
                gen_layer, train_layer, hidden, train_config.num_moe_experts
            )
            if torch.distributed.get_rank() == 0:
                print(
                    f"[mega-metric] routing diff (train vs gen, {count} tokens): "
                    f"{sets} tokens pick a different expert set, {orders} differ in "
                    f"order only, max prob gap={gap:.3e}",
                    flush=True,
                )
            results.append((count, error, differed))

        # Asserted after the loop so every count is measured. Whether the
        # affected tokens scale with the batch (a per-token rate) or stay at one
        # (a specific token) is the distinguishing fact, and stopping at the
        # first failure would hide it.
        worst = max(result[1] for result in results)
        assert worst < 1e-6, (
            "training and generation disagree on "
            + ", ".join(f"{differed}/{count} tokens" for count, _, differed in results)
            + f"; worst rel_rms={worst:.3e}. Batch size is not the variable, since "
            "test_generation_is_batch_invariant is bitwise at these counts, so read "
            "the routing diff metric above."
        )

    def test_generation_is_batch_invariant(self):
        """Attribution for the token-count result: is batch size FlashInfer's?

        One generation layer, two batchings of the same tokens. Nothing of ours
        participates -- no repack, no shared scratch, no passthrough dispatcher,
        no dense-to-topk conversion -- and it is the same instance both times,
        so the per-construction flake seen elsewhere cannot confound it either.
        The only difference is how many tokens share a launch.

        Read against ``test_parity_holds_across_token_counts``: if both move,
        batch invariance is a kernel property and our training path is not
        implicated, which also means chunking the training forward is the fix.
        If this is clean while train-vs-gen is not, something in the training
        path is batch-dependent and chunking would only mask it.
        """
        capacity = max(GEN_TOKENS, *TRAIN_TOKEN_COUNTS)
        config = _config(mega_training=False, inference_mega_max_tokens_per_rank=capacity)
        gen_layer = _build_layer(config, for_inference=True).eval()

        for count in TRAIN_TOKEN_COUNTS:
            hidden = _hidden(config, seed=5, tokens=count)
            with torch.no_grad(), InferenceMode.active():
                batched, _ = gen_layer(hidden)
            chunked = _chunked_generation(gen_layer, hidden, GEN_TOKENS)
            error, differed = _measure_token_count_parity(batched, chunked, "gen vs gen")

            assert error < 1e-6, (
                f"the kernel is not batch-invariant: {differed}/{count} tokens differ "
                f"between one {count}-token forward and the same tokens taken "
                f"{GEN_TOKENS} at a time, rel_rms={error:.3e}; the generation path "
                "alone is involved on both sides"
            )

    def test_two_identically_built_generation_layers_agree(self):
        """Attribution: does the disagreement need any of our code at all?

        Both layers take the pre-existing inference path -- FlashInfer's own
        weight pack and ``preprocess_weights=True`` -- so none of the training
        work is involved: no repack, no shared scratch, no passthrough
        dispatcher, no dense-to-topk routing conversion. A disagreement here
        reproduces the effect with nothing of ours in the picture.

        This is the arm ``test_two_identically_built_layers_agree`` cannot
        supply. Those two layers repack into the one shared scratch, so their
        weight tensors sit at the same address with the same strides, which
        makes them blind to an allocation-dependent effect -- and since the two
        weight paths are known to produce bitwise-identical bytes, that is the
        hypothesis left. FlashInfer allocates a fresh transformed tensor per
        construction, so these two do not share an address.

        Built in a loop because the rate is a few percent per construction: one
        comparison per launch would need far more launches to say anything.
        """
        config = _config(mega_training=False)
        hidden = _hidden(config, seed=1)

        errors = []
        for _ in range(PARITY_BUILDS):
            model_parallel_cuda_manual_seed(123)
            first = _build_layer(config, for_inference=True).eval()
            second = _build_layer(config, for_inference=True).eval()
            _copy_expert_weights(first, second)
            with torch.no_grad(), InferenceMode.active():
                out_first, _ = first(hidden)
                out_second, _ = second(hidden)
            errors.append(_rel_rms(out_second, out_first))

        stats = torch.tensor(
            [max(errors), float(sum(e != 0.0 for e in errors))], device="cuda"
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
        worst, differed = stats[0].item(), int(stats[1].item())
        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] two identically built gen layers: worst={worst:.3e} "
                f"({differed}/{PARITY_BUILDS} pairs differed)",
                flush=True,
            )

        assert worst < 1e-6, (
            f"{differed}/{PARITY_BUILDS} pairs of identically built generation "
            f"layers disagree, worst rel_rms={worst:.3e}; no Megatron repack is "
            "involved on either side"
        )

    def test_two_identically_built_layers_agree(self):
        """Separates kernel nondeterminism from the two weight paths.

        Both instances are built the same way, so the caller-owned repack is on
        both sides. Note the blind spot: they share the one process-wide
        scratch, so their weights sit at the same address and this cannot detect
        an allocation-dependent difference. Kept as the train-path counterpart
        to the generation-path test above, not as evidence on its own.
        """
        config = _config(mega_training=True)
        first = _build_layer(config).train()
        second = _build_layer(config).train()
        _copy_expert_weights(first, second)

        # Sequential by necessity, not by accident: both layers alias the one
        # shared scratch, and each forward repacks and takes ownership before
        # computing, so the outputs must be taken one after the other.
        hidden = _hidden(config, seed=1)
        out_first, _ = first(hidden)
        out_second, _ = second(hidden)

        error = _measure(out_second, out_first, "two identically built layers")
        assert error < 1e-6, f"identically built layers disagree: rel_rms={error:.3e}"

    def test_parity_is_stable_across_layer_constructions(self):
        """Whether a freshly built layer pair agrees is itself the variable.

        Repeated forwards on one pair agree bitwise, yet separately built pairs
        have disagreed at ~1e-4 from one run to the next with no code change. So
        the weights and the input are reseeded identically for every pair here,
        leaving construction as the only thing that differs; a mix of zero and
        nonzero results then pins the cause to per-instance state rather than to
        anything weight-dependent.

        The pairs are not freed as they go: FlashInfer tears the megakernel
        workspace down with an EP collective, so releasing mid-loop would add a
        new way to desynchronize. The fixture handles it, as in the other tests.
        """
        gen_config = _config(mega_training=False)
        train_config = _config(mega_training=True)
        hidden = _hidden(train_config, seed=1)

        errors = []
        for _ in range(PARITY_BUILDS):
            # Without reseeding, the RNG hands each pair different weights, which
            # would confound a weight-dependent difference with a per-build one.
            model_parallel_cuda_manual_seed(123)
            gen_layer = _build_layer(gen_config, for_inference=True).eval()
            train_layer = _build_layer(train_config).train()
            _copy_expert_weights(gen_layer, train_layer)

            with torch.no_grad(), InferenceMode.active():
                gen_out, _ = gen_layer(hidden)
            train_out, _ = train_layer(hidden)
            errors.append(_rel_rms(train_out, gen_out))

        stats = torch.tensor(
            [max(errors), float(sum(e != 0.0 for e in errors))], device="cuda"
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
        worst, differed = stats[0].item(), int(stats[1].item())
        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] train/gen parity across builds: worst={worst:.3e} "
                f"({differed}/{PARITY_BUILDS} pairs differed)",
                flush=True,
            )

        assert worst < 1e-6, (
            f"{differed}/{PARITY_BUILDS} freshly built layer pairs disagree, "
            f"worst rel_rms={worst:.3e}"
        )

    def test_parity_holds_after_a_weight_update(self):
        """Guards the repack: a snapshotted kernel weight would drift here."""
        config = _config(mega_training=True)
        gen_layer = _build_layer(_config(mega_training=False), for_inference=True).eval()
        train_layer = _build_layer(config).train()
        _copy_expert_weights(gen_layer, train_layer)

        hidden = _hidden(config, seed=2)
        train_layer(hidden)

        # Stand in for an optimizer step, applied to both layers.
        for layer in (gen_layer, train_layer):
            _bump_expert_weights(layer)

        with torch.no_grad(), InferenceMode.active():
            gen_out, _ = gen_layer(hidden)
        train_out, _ = train_layer(hidden)

        error = _measure(train_out, gen_out, "train/gen parity after update")
        assert error < 1e-6, f"parity lost after weight update: rel_rms={error:.3e}"


class TestGenerationWeightOwnership:
    """(4) Generation serves the weights it was last refit with.

    The megakernel holds its own transformed copy of the expert weights and
    reads that copy, not the parameters. There are two ways to keep it current:
    let FlashInfer preprocess the parameters and snapshot the result, which is
    right exactly once, or own the buffer and rewrite it in place. bf16 does the
    latter so that an RL refit is a repack rather than a teardown -- tearing a
    mega layer down and rebuilding it costs an EP collective, a symmetric-heap
    reallocation and a CuTeDSL recompile per refit.

    These cover what that buys and what it risks: that the repack reproduces
    what FlashInfer would have produced, that a refit actually reaches the
    kernel, and that one layer's repack cannot be served to another.
    """

    def test_caller_owned_weights_match_flashinfer_preprocessing(self):
        """The repack is equivalent to the path it replaced, end to end.

        ``test_mega_training_weights.py`` already pins the packed bytes against
        FlashInfer's own ``_interleave_gate_up_32``. This is the other half:
        that handing those bytes over as ``transformed_weights`` with
        ``preprocess_weights=False`` produces the same forward as letting
        FlashInfer build the pack itself, so the adapter wiring is right and not
        just the layout.
        """
        config = _config(mega_training=False)
        owned = _build_layer(config, for_inference=True).eval()
        preprocessed = _build_layer(config, for_inference=True).eval()
        _copy_expert_weights(owned, preprocessed)
        _use_flashinfer_preprocessing(preprocessed)

        hidden = _hidden(config, seed=6)
        with torch.no_grad(), InferenceMode.active():
            owned_out, _ = owned(hidden)
            preprocessed_out, _ = preprocessed(hidden)

        error = _measure(owned_out, preprocessed_out, "caller-owned vs preprocessed weights")
        assert error < 1e-6, (
            f"the caller-owned repack does not reproduce FlashInfer's own weight "
            f"preprocessing: rel_rms={error:.3e}"
        )

    def test_refit_without_a_refresh_serves_stale_weights(self):
        """Negative control: the staleness the refit hook exists to prevent is real.

        Writing the parameters is not enough on its own, which is the whole
        reason ``resharding.refit`` has to call the hook. Worth asserting rather
        than assuming, because the failure it guards against is silent -- the
        rollouts stay finite and plausible, they just come from the previous
        step's policy, and the importance ratio that is supposed to catch train/
        gen mismatch is computed against those same stale log-probs.
        """
        config = _config(mega_training=False)
        layer = _build_layer(config, for_inference=True).eval()
        hidden = _hidden(config, seed=6)

        with torch.no_grad(), InferenceMode.active():
            before, _ = layer(hidden)
            # Cloned because FlashInfer may hand back a workspace tensor it
            # overwrites on the next call, which would make the comparison
            # below pass by aliasing rather than by staleness.
            before = before.clone()

        _bump_expert_weights(layer)
        with torch.no_grad(), InferenceMode.active():
            stale, _ = layer(hidden)
        assert torch.equal(before, stale), (
            "the kernel picked up a parameter write with no refresh, so the "
            "buffer is not actually owned and cached; this test no longer "
            "controls for anything"
        )

        assert layer.experts.refresh_mega_weights() is True
        with torch.no_grad(), InferenceMode.active():
            refreshed, _ = layer(hidden)
        assert not torch.equal(before, refreshed), (
            "refresh_mega_weights() did not reach the kernel: generation still "
            "returns the pre-refit output"
        )

    def test_refresh_after_a_refit_matches_a_freshly_built_layer(self):
        """The refreshed weights are right, not merely different.

        A refit is only correct if the refitted layer becomes
        indistinguishable from a layer that had the new weights all along, which
        is the reference here. Bitwise, since both sides pack the same bytes and
        run the same kernel.
        """
        config = _config(mega_training=False)
        refitted = _build_layer(config, for_inference=True).eval()
        hidden = _hidden(config, seed=6)

        # The first forward is what binds the kernel to the pre-refit weights;
        # without it there would be nothing stale to refresh.
        with torch.no_grad(), InferenceMode.active():
            refitted(hidden)
        _bump_expert_weights(refitted)
        assert refitted.experts.refresh_mega_weights() is True

        reference = _build_layer(config, for_inference=True).eval()
        _copy_expert_weights(refitted, reference)

        with torch.no_grad(), InferenceMode.active():
            refitted_out, _ = refitted(hidden)
            reference_out, _ = reference(hidden)

        error = _measure(refitted_out, reference_out, "gen weights after refit")
        assert error < 1e-6, (
            f"a refreshed layer and a freshly built one disagree: rel_rms={error:.3e}; "
            "the repack ran but did not reproduce the refit weights"
        )

    def test_refit_discovers_the_refresh_hook(self):
        """``resharding.refit`` finds the hook by name, so keep it findable.

        The refit walks ``tgt_core.modules()`` and calls whatever answers to
        ``refresh_mega_weights``. Nothing type-checks that, so renaming the
        method or moving it off the experts module would disable refit silently
        and leave generation on snapshotted weights.
        """
        config = _config(mega_training=False)
        layer = _build_layer(config)

        hooks = [
            module
            for module in layer.modules()
            if getattr(module, "refresh_mega_weights", None) is not None
        ]
        assert hooks == [layer.experts], (
            "refresh_mega_weights is not reachable from the module tree the way "
            f"resharding.refit looks it up; found {hooks}"
        )
        assert hooks[0].refresh_mega_weights() is True

    def test_each_generation_layer_owns_its_weight_buffer(self):
        """Generation buffers are per layer, unlike the training scratch.

        The training forward shares one buffer across the whole model because it
        repacks inside every layer's forward. Generation cannot: it packs once
        and reuses, so a shared buffer would mean every layer but the last
        computed with whichever layer packed most recently -- and a real model
        has dozens of MoE layers, while every other test here has one.
        """
        config = _config(mega_training=False)
        first = _build_layer(config, for_inference=True).eval()
        second = _build_layer(config, for_inference=True).eval()
        # Deliberately not synced: identical experts would make a shared buffer
        # undetectable. Successive builds draw different weights from the RNG,
        # which the first assertion below confirms.
        hidden = _hidden(config, seed=6)

        with torch.no_grad(), InferenceMode.active():
            first_before, _ = first(hidden)
            first_before = first_before.clone()
            second_out, _ = second(hidden)
            second_out = second_out.clone()
            first_after, _ = first(hidden)

        assert not torch.equal(first_before, second_out), (
            "the two layers hold the same expert weights, so this cannot detect "
            "a shared buffer"
        )
        assert torch.equal(first_before, first_after), (
            "one generation layer's output changed after another layer ran, so "
            "they are sharing a weight buffer"
        )

    def test_refresh_refuses_a_quantized_precision(self):
        """The quantized precisions cannot be refit, and say so rather than drifting.

        FlashInfer quantizes while it preprocesses; the repack only reshapes, so
        there is no way to rebuild an mxfp8 or nvfp4 kernel weight from the bf16
        parameters. Refusing is the point -- the alternative is generation
        quietly continuing on the weights snapshotted before the refit. See
        ``megatron/core/inference/moe/mega/QUANTIZED_BLOCKERS.md``.
        """
        config = _config(mega_training=False)
        layer = _build_layer(config)
        # The precision is read at refresh time rather than captured at
        # construction, so flipping it here reaches the guard without needing
        # quantized parameters -- which is convenient, since this refusal is
        # itself why no quantized RL path exists to build them.
        layer.experts.config.inference_mega_precision = "mxfp8"

        with pytest.raises(NotImplementedError, match="quantizes the weights"):
            layer.experts.refresh_mega_weights()


class TestAgainstStandardBf16:
    """(3) Forward and backward stay close to the standard TE bf16 path."""

    # The mega kernel reduces the expert combine in fp32 in one pass while the TE
    # path accumulates in bf16 across dispatch and combine, so the two differ by
    # reduction order rather than by algorithm. These bound that difference; they
    # are not bitwise claims.
    #
    # bf16 keeps 8 mantissa bits, so a single rounding is ~2e-3 relative, and the
    # two paths round at different points: the fc1 output, the SwiGLU product, the
    # fc2 output, and the combine. A few of those compound to the ~6e-3 actually
    # observed at topk=8 (~5e-3 at topk=2, the combine being one term of it).
    # The bound keeps headroom for seed and token-count variation
    # while staying far below the O(0.1) error that a wrong expert, a wrong
    # gate/up interleave, or divergent routing produces, which is what this is
    # meant to catch. The tight numbers live in TestTrainGenParity.
    FORWARD_TOL = 1.5e-2
    GRAD_TOL = 5e-2

    def test_forward_matches_te_bf16(self):
        config = _config(mega_training=True)
        te_layer = _build_layer(_config(mega_training=False)).train()
        mega_layer = _build_layer(config).train()
        _copy_expert_weights(te_layer, mega_layer)

        hidden = _hidden(config, seed=3)
        te_out, _ = te_layer(hidden)
        mega_out, _ = mega_layer(hidden)

        error = _measure(mega_out, te_out, "mega vs TE bf16 forward")
        assert error < self.FORWARD_TOL, f"forward diverges from TE bf16: rel_rms={error:.3e}"

    def test_input_and_weight_gradients_match_te_bf16(self):
        config = _config(mega_training=True)
        te_layer = _build_layer(_config(mega_training=False)).train()
        mega_layer = _build_layer(config).train()
        _copy_expert_weights(te_layer, mega_layer)

        te_hidden = _hidden(config, seed=4).requires_grad_(True)
        mega_hidden = te_hidden.detach().clone().requires_grad_(True)

        te_out, _ = te_layer(te_hidden)
        mega_out, _ = mega_layer(mega_hidden)
        # A fixed non-uniform weighting, so the reduction does not mask
        # per-position differences the way sum() would.
        grad_out = torch.randn(
            te_out.shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator(device="cuda").manual_seed(5),
        )
        te_out.backward(grad_out)
        mega_out.backward(grad_out)

        dgrad_error = _measure(mega_hidden.grad, te_hidden.grad, "mega vs TE bf16 dgrad")
        assert dgrad_error < self.GRAD_TOL, f"dgrad diverges: rel_rms={dgrad_error:.3e}"

        wgrad_error = _measure(
            mega_layer.experts.linear_fc1.weight0.grad,
            te_layer.experts.linear_fc1.weight0.grad,
            "mega vs TE bf16 wgrad",
        )
        assert wgrad_error < self.GRAD_TOL, f"wgrad diverges: rel_rms={wgrad_error:.3e}"


class TestWholeTransformerLayer:
    """(4) The MoE layer inside the transformer layer it actually ships in.

    Turning on moe_mega_training_forward also forces the training side onto
    transformer_impl='inference_optimized', so attention, the projections and
    the norms all change implementation along with the experts. The MoE-only
    tests above cannot see any of that, and a first RL run found its failures
    in exactly that surrounding code.

    Generation is deliberately absent: it needs a dynamic inference context and
    a KV cache to compare against, which is the end-to-end run's job.
    """

    # Looser than the MoE-only comparison because a whole layer puts attention,
    # two norms and the projections in front of the expert output, so the same
    # kernel difference arrives having been through more arithmetic.
    BLOCK_TOL = 2e-2
    SEQ_LEN = 16
    BATCH = 2

    def _config(self, mega_training: bool):
        # SEQ_LEN * BATCH tokens per rank has to stay under the workspace cap
        # that _config sets, and hidden_dropout is off so train and eval differ
        # only in which kernel runs, not in sampled noise.
        return _config(mega_training, hidden_dropout=0.0)

    def _build_block(self, config):
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_inference_submodules,
        )
        from megatron.core.transformer.transformer_layer import TransformerLayer

        submodules = get_gpt_layer_with_inference_submodules(
            num_experts=config.num_moe_experts, moe_grouped_gemm=True
        )
        return TransformerLayer(config, submodules).cuda()

    def _inputs(self, config, seed):
        hidden = torch.randn(
            (self.SEQ_LEN, self.BATCH, config.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        mask = torch.ones((1, 1, self.SEQ_LEN, self.SEQ_LEN), dtype=bool, device="cuda")
        return hidden, mask

    @staticmethod
    def _count_mega_calls(block):
        """Count entries into the megakernel.

        Without this a fallback to TE would leave every assertion here still
        passing, since the two paths agree to within the tolerances being
        checked -- the test would go green while measuring nothing.
        """
        experts = block.mlp.experts
        calls = []
        original = experts._mega_training_forward_pass

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        experts._mega_training_forward_pass = counting
        return calls

    def test_layer_forward_backward_runs(self):
        config = self._config(mega_training=True)
        block = self._build_block(config).train()
        calls = self._count_mega_calls(block)

        hidden, mask = self._inputs(config, seed=11)
        hidden.requires_grad_(True)
        out, _ = block(hidden_states=hidden, attention_mask=mask)
        out.sum().backward()

        assert out.shape == (self.SEQ_LEN, self.BATCH, config.hidden_size)
        assert torch.isfinite(out).all(), "non-finite activations out of the layer"
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert len(calls) == 1, (
            f"expected exactly one megakernel call in the value pass, got {len(calls)}. "
            "0 means the layer fell back to TE; >1 means the recompute pass took "
            "the mega path too, which would leave the backward without a graph."
        )

    def test_eval_forward_matches_train_forward(self):
        """Log-prob shape of the pass, inside the full layer.

        With dropout off the two modes should produce the same value, and both
        should reach the kernel. Before the eval-mode fix this failed with a
        difference around the TE-vs-mega gap rather than zero.
        """
        config = self._config(mega_training=True)
        block = self._build_block(config)
        hidden, mask = self._inputs(config, seed=12)

        block.train()
        with torch.no_grad():
            train_out, _ = block(hidden_states=hidden, attention_mask=mask)

        block.eval()
        eval_calls = self._count_mega_calls(block)
        with torch.no_grad():
            eval_out, _ = block(hidden_states=hidden, attention_mask=mask)

        assert len(eval_calls) == 1, (
            f"eval-mode forward made {len(eval_calls)} megakernel calls, expected 1. "
            "0 means the log-prob pass fell back to TE."
        )
        error = _measure(eval_out, train_out, "whole layer: eval vs train forward")
        assert error < 1e-6, f"eval and train forwards disagree: rel_rms={error:.3e}"

    def test_layer_forward_matches_te_layer(self):
        config = self._config(mega_training=True)
        te_block = self._build_block(self._config(mega_training=False)).train()
        mega_block = self._build_block(config).train()
        # Whole-layer copy, not just the experts: attention and the norms have
        # to match too or the comparison measures initialization, not kernels.
        mega_block.load_state_dict(te_block.state_dict())

        hidden, mask = self._inputs(config, seed=13)
        te_out, _ = te_block(hidden_states=hidden, attention_mask=mask)
        mega_out, _ = mega_block(hidden_states=hidden, attention_mask=mask)

        error = _measure(mega_out, te_out, "whole layer: mega vs TE")
        assert error < self.BLOCK_TOL, f"layer output diverges from TE: rel_rms={error:.3e}"


def _capture_routing(layer, for_inference: bool):
    """Record the routing indices this layer hands the kernel.

    Both paths funnel through ``adapter.forward(hidden, routing_map, probs, ...)``,
    so wrapping it captures exactly what the kernel selects -- after
    InferenceTopKRouter's topk on one side and after
    ``_mega_dense_routing_to_topk`` recovers the set from TopKRouter's boolean
    map on the other. Comparing anything earlier would compare two different
    representations; comparing the outputs alone cannot say whether a
    disagreement is a different expert or different arithmetic.
    """
    experts = layer.experts
    adapter = experts._mega_adapter if for_inference else experts._mega_training_adapter
    captured = []
    original = adapter.forward

    def capturing(hidden_states, routing_map, probs, *args, **kwargs):
        captured.append(routing_map.detach().clone())
        return original(hidden_states, routing_map, probs, *args, **kwargs)

    adapter.forward = capturing
    return captured


def _routing_disagreements(gen_indices, train_indices):
    """Count tokens whose selected expert *set* differs.

    Sets, not sequences: order is known not to matter -- the kernel's combine
    was bitwise invariant under permutation at topk=8 -- so comparing the
    indices elementwise would report differences the output cannot see, and
    hide nothing that it can.
    """
    gen_sorted = gen_indices.sort(dim=-1).values
    train_sorted = train_indices.sort(dim=-1).values
    per_token = (gen_sorted != train_sorted).any(dim=-1)
    return int(per_token.sum().item()), per_token


def _perturb_router(layer, scale, seed):
    """Nudge the router, the way an optimizer step does and a uniform bump does not.

    ``_bump_expert_weights`` adds one delta to every expert, which leaves every
    router score shifted by the same amount and so cannot change what the top-k
    selects. A step-2 divergence needs the *relative* order of scores to move,
    which takes per-element noise on the router itself.
    """
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "router" in name:
                generator = torch.Generator(device=param.device).manual_seed(seed)
                noise = torch.randn(
                    param.shape, device=param.device, dtype=param.dtype, generator=generator
                )
                param.add_(noise * scale)


def _tie_the_router(layer):
    """Make experts tie exactly, in pairs.

    A near-tie is what the e2e run is suspected of hitting, and waiting for one
    to appear by chance is a test that passes for the wrong reason. Duplicating
    router rows makes every even/odd expert pair score identically for every
    token, so the top-k boundary is ambiguous on purpose and the two
    implementations have to break it the same way to agree.
    """
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "router" in name and param.dim() == 2:
                param[1::2] = param[0::2]


class TestParityAfterARouterUpdate:
    """(6) The step-2 gap from the RL pipeclean, brought down to one layer.

    The run scored 2048/2048 tokens bitwise equal at step 1 and lost 125 of
    them at step 2 -- concentrated in 2 of 8 sequences, with an onset point and
    a worst case of 0.26 nats, which is the size of a token being sent to a
    different expert rather than of arithmetic drift.

    Nothing above can see that. Every parity test runs on freshly initialized
    weights, where router scores are well separated, and the one test that does
    perturb weights adds a single delta to every expert -- which shifts all
    router scores equally and cannot reorder them. These reproduce the two
    candidate mechanisms directly: a selection that disagrees once scores move,
    and a selection that disagrees when scores tie.
    """

    # Wide enough that a per-token event of order 1e-3 is likely to appear at
    # least once, which the 8-token default cannot do.
    TOKENS = int(os.environ.get("MEGA_TEST_UPDATE_TOKENS", "2048"))
    # Small next to the initialization, large next to an ULP: the regime an
    # optimizer step leaves the router in after a handful of updates.
    NOISE = float(os.environ.get("MEGA_TEST_ROUTER_NOISE", "0.02"))

    def _layers(self):
        """A generation layer and a training layer with bit-identical weights."""
        capacity = self.TOKENS
        gen = _build_layer(
            _config(mega_training=False, inference_mega_max_tokens_per_rank=capacity),
            for_inference=True,
        )
        train = _build_layer(
            _config(mega_training=True, inference_mega_max_tokens_per_rank=capacity)
        ).train()
        return gen, train

    def _run_both(self, gen, train, hidden):
        gen_routing = _capture_routing(gen, for_inference=True)
        train_routing = _capture_routing(train, for_inference=False)
        with torch.no_grad():
            with InferenceMode.active():
                gen_out, _ = gen(hidden)
            train_out, _ = train(hidden)
        assert gen_routing and train_routing, "adapter was not reached; mega path inactive"
        return gen_out, train_out, gen_routing[0], train_routing[0]

    def test_routing_and_output_agree_after_a_router_update(self):
        """Perturb the router on both sides identically, then compare."""
        gen, train = self._layers()
        _perturb_router(train, scale=self.NOISE, seed=17)
        # Copied after the perturbation, so both sides hold the same bits. A
        # difference here would be the test's own doing, not the kernel's.
        _copy_expert_weights(train, gen)

        hidden = _hidden(train.config, seed=18, tokens=self.TOKENS)
        gen_out, train_out, gen_idx, train_idx = self._run_both(gen, train, hidden)

        disagreed, mask = _routing_disagreements(gen_idx, train_idx)
        error = _rel_rms(train_out, gen_out)
        stats = torch.tensor([disagreed, error], device="cuda", dtype=torch.float64)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
        disagreed, error = int(stats[0].item()), stats[1].item()

        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] post-update routing: {disagreed}/{self.TOKENS} tokens "
                f"disagree, output rel_rms={error:.3e}",
                flush=True,
            )
        assert disagreed == 0, (
            f"{disagreed}/{self.TOKENS} tokens routed to different experts after a "
            "router update. This is the step-2 e2e divergence, reproduced: the two "
            "top-k implementations disagree once scores are no longer well separated."
        )
        assert error < 1e-6, (
            f"output differs after a router update despite identical routing: "
            f"rel_rms={error:.3e}. Routing agrees, so this is the kernel, not selection."
        )

    def test_routing_agrees_when_router_scores_tie(self):
        """Force exact ties rather than waiting for one to occur by chance."""
        gen, train = self._layers()
        _tie_the_router(train)
        _copy_expert_weights(train, gen)

        hidden = _hidden(train.config, seed=19, tokens=self.TOKENS)
        _, _, gen_idx, train_idx = self._run_both(gen, train, hidden)

        disagreed, _ = _routing_disagreements(gen_idx, train_idx)
        stats = torch.tensor([disagreed], device="cuda", dtype=torch.float64)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
        disagreed = int(stats[0].item())

        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] tied-score routing: {disagreed}/{self.TOKENS} tokens disagree",
                flush=True,
            )
        assert disagreed == 0, (
            f"{disagreed}/{self.TOKENS} tokens routed differently when experts tie. "
            "TopKRouter's boolean map and InferenceTopKRouter's topk break exact ties "
            "differently; the two paths must agree by construction, not by luck."
        )

    def test_batched_and_chunked_generation_agree_after_an_update(self):
        """The other candidate: token grouping, in the post-update regime.

        Generation saw sequence 3 a few decode tokens at a time while the
        log-prob pass saw all 512 at once. Batch invariance was only ever
        checked on unperturbed weights, where routing is not near a boundary.
        """
        gen, train = self._layers()
        _perturb_router(train, scale=self.NOISE, seed=20)
        _copy_expert_weights(train, gen)

        hidden = _hidden(train.config, seed=21, tokens=self.TOKENS)
        with torch.no_grad(), InferenceMode.active():
            batched, _ = gen(hidden)
        # GEN_TOKENS, not a literal: the shared helper prints that constant as the
        # chunk width, so any other value makes the metric line describe a run
        # that did not happen.
        chunked = _chunked_generation(gen, hidden, chunk=GEN_TOKENS)

        error, differed = _measure_token_count_parity(
            batched, chunked, f"post-update gen batched vs chunked({GEN_TOKENS})"
        )
        # Asserting on the token count as well as the rms: an rms over 2048
        # tokens dilutes the handful of badly wrong ones this is looking for,
        # which is exactly the shape the e2e run showed.
        assert differed == 0 and error < 1e-6, (
            f"generation depends on token grouping after a router update: "
            f"{differed}/{self.TOKENS} tokens differ, rel_rms={error:.3e}. Decode-sized "
            "launches and one wide launch disagree, which is the other way the e2e "
            "log-prob pass can diverge from the rollout."
        )


class TestRefitReachesTheKernelAfterGenerationHasRun:
    """(7) The refit ordering the e2e run has and the tests above do not.

    The pipeclean scored 0 KL at step 1 and 8e-5 at step 2, against 1e-14 for
    the TE arm on the same recipe. Step 1 is the step at which no optimizer
    step has happened yet, which is exactly when serving stale weights is
    indistinguishable from serving correct ones, so "generation is one update
    behind" fits the shape of that result.

    ``InferenceGroupedMLP`` may copy the per-expert parameters into one
    contiguous ``_fc1_weight`` on first use and repoint ``param.data`` at views
    of it. A refit resolving its destination through the ``nn.Parameter``
    follows that redirect; one holding the pre-redirect tensor writes to
    storage nothing reads any more. RL's first refit runs before the first
    rollout, so its plan -- cached and reused -- is built while the parameters
    still own their original storage.

    The reference here is built from the weights the refit *intended* to
    install, not from the layer's own state afterwards. Deriving it from the
    layer is how the first version of this test reported agreement in both
    arms: a write that never landed leaves reference and layer equally stale.
    """

    BUMP = 0.05

    def _gen_layer(self):
        return _build_layer(_config(mega_training=False), for_inference=True)

    def _expert_params(self, experts):
        return [
            getattr(experts.linear_fc1, f"weight{i}")
            for i in range(experts.num_local_experts)
        ]

    def _intended_output(self, snapshot, hidden):
        """A layer holding exactly the weights the refit meant to install.

        Built and bumped before its own first forward, while the parameters
        still own their storage, so this arm cannot be affected by the
        redirect it is being used to detect.
        """
        reference = self._gen_layer()
        reference.load_state_dict(snapshot)
        with torch.no_grad():
            for param in self._expert_params(reference.experts):
                param.data.add_(self.BUMP)
        with torch.no_grad(), InferenceMode.active():
            out, _ = reference(hidden)
        return out

    def _refit_and_compare(self, write_through_parameter: bool):
        layer = self._gen_layer()
        experts = layer.experts
        hidden = _hidden(layer.config, seed=31)
        snapshot = {k: v.clone() for k, v in layer.state_dict().items()}

        # Captured before any forward, which is when RL's plan is built: the
        # first refit precedes the first rollout.
        params = self._expert_params(experts)
        pre_redirect = [p.data for p in params]
        pre_ptrs = [t.data_ptr() for t in pre_redirect]

        # The rollout, which is what would trigger the redirect.
        with torch.no_grad(), InferenceMode.active():
            layer(hidden)

        redirected = [p.data_ptr() for p in params] != pre_ptrs
        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] rollout redirected param.data: {redirected} "
                f"(_fc1_weight buffer: {hasattr(experts, '_fc1_weight')})",
                flush=True,
            )

        with torch.no_grad():
            for param, pre in zip(params, pre_redirect):
                (param.data if write_through_parameter else pre).add_(self.BUMP)

        refresh = getattr(experts, "refresh_mega_weights", None)
        assert refresh is not None, "refresh_mega_weights missing; refit cannot reach the kernel"
        refresh()

        with torch.no_grad(), InferenceMode.active():
            after, _ = layer(hidden)
        return _rel_rms(after, self._intended_output(snapshot, hidden)), redirected

    def test_refit_through_the_parameter_reaches_the_kernel(self):
        """The good case, and the one the e2e path needs to be taking."""
        error, _ = self._refit_and_compare(write_through_parameter=True)
        error = _measure_scalar(error, "refit via nn.Parameter after a rollout")
        assert error < 1e-6, (
            f"a refit written through param.data after generation has run does not "
            f"reach the kernel: rel_rms={error:.3e}. Generation would serve the "
            f"previous step's expert weights, which is the e2e signature."
        )

    def test_a_refit_holding_pre_redirect_storage_would_be_caught(self):
        """Whether the pre-redirect tensor is still the one the kernel reads.

        This is diagnostic rather than a correctness requirement: it decides
        whether a cached plan built before the first rollout can go stale. If
        the rollout never redirects ``param.data`` the question is moot, and
        the test says so instead of asserting something that does not apply.
        """
        error, redirected = self._refit_and_compare(write_through_parameter=False)
        error = _measure_scalar(error, "refit via pre-redirect storage after a rollout")
        if not redirected:
            pytest.skip(
                "the rollout does not repoint param.data, so a plan built before it "
                "cannot hold orphaned storage and this cannot explain the e2e gap"
            )
        assert error > 1e-6, (
            "param.data was repointed by the rollout, yet writing to the pre-redirect "
            "tensor still reached the kernel. That combination is not expected; "
            "re-read the redirect before trusting either arm of this class."
        )


def _boundary_gap_stats(layer, hidden):
    """How often the top-k boundary is too close to call.

    Two experts whose router scores differ by less than the score's own
    representation can resolve are a coin flip, and a flip sends the token to a
    different expert. The gap between the k-th and (k+1)-th score is that coin.

    This is the quantity that scales with expert count, and the reason the
    tests above cannot speak to the e2e run: top-8-of-16 puts the boundary at
    the median of the score distribution, where scores are far apart, while
    top-8-of-128 puts it in the packed tail. Same topk, different regime.
    """
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = F.linear(flat.float(), layer.router.weight.float())
    scores = torch.softmax(logits, dim=-1)
    top = scores.topk(ROUTER_TOPK + 1, dim=-1).values
    relative = (top[:, -2] - top[:, -1]) / top[:, -2].abs().clamp_min(1e-30)
    # bf16 carries 8 mantissa bits; below that the two scores are the same number.
    return int((relative <= 2.0**-8).sum().item()), relative.numel(), relative


class TestRoutingAtTheProductionExpertCount:
    """(8) The same parity question, at the expert count the e2e run uses.

    Every test above runs 16 experts because it is fast, and that choice turns
    out to decide the answer rather than just the runtime. The e2e model routes
    top-8 of 128. The suspected mechanism -- two experts scoring close enough
    at the selection boundary that the two paths can order them differently --
    has a rate that depends entirely on how crowded that boundary is, so a
    16-expert test can report zero disagreements no matter whether the
    mechanism is real.

    One build, many cheap trials: the layer construction is what costs seconds,
    while a fresh hidden state and two forwards cost milliseconds, so the token
    count this samples is set by the loop rather than by the clock.

    Reports the boundary-gap density alongside the disagreement count. If the
    density here is far below what 128 experts produce in the real model, a
    clean result means the test still has not reached the regime, and saying so
    is more useful than the pass.
    """

    EXPERTS = int(os.environ.get("MEGA_TEST_PROD_EXPERTS", "128"))
    TOKENS = int(os.environ.get("MEGA_TEST_PROD_TOKENS", "2048"))
    TRIALS = int(os.environ.get("MEGA_TEST_PROD_TRIALS", "6"))
    NOISE = float(os.environ.get("MEGA_TEST_ROUTER_NOISE", "0.02"))

    def test_routing_agrees_when_the_boundary_is_crowded(self):
        if self.EXPERTS % EP_SIZE:
            pytest.skip(f"{self.EXPERTS} experts do not divide EP={EP_SIZE}")

        shape = dict(
            num_moe_experts=self.EXPERTS,
            inference_mega_max_tokens_per_rank=self.TOKENS,
        )
        gen = _build_layer(_config(mega_training=False, **shape), for_inference=True)
        train = _build_layer(_config(mega_training=True, **shape)).train()

        disagreed_total = 0
        worst_output = 0.0
        unresolvable = 0
        boundary_total = 0
        medians = []

        for trial in range(self.TRIALS):
            # A fresh router draw per trial, then copied across, so each trial
            # is an independent sample of the boundary rather than the same
            # weights seen again with new inputs.
            _perturb_router(train, scale=self.NOISE, seed=1000 + trial)
            _copy_expert_weights(train, gen)
            hidden = _hidden(train.config, seed=2000 + trial, tokens=self.TOKENS)

            gen_adapter = gen.experts._mega_adapter
            train_adapter = train.experts._mega_training_adapter
            gen_saved, train_saved = gen_adapter.forward, train_adapter.forward
            gen_seen, train_seen = [], []

            def capture(original, sink):
                def wrapper(hidden_states, routing_map, probs, *args, **kwargs):
                    sink.append(routing_map.detach().clone())
                    return original(hidden_states, routing_map, probs, *args, **kwargs)

                return wrapper

            gen_adapter.forward = capture(gen_saved, gen_seen)
            train_adapter.forward = capture(train_saved, train_seen)
            try:
                with torch.no_grad():
                    with InferenceMode.active():
                        gen_out, _ = gen(hidden)
                    train_out, _ = train(hidden)
            finally:
                # Restored every trial; leaving the wrappers in place would
                # stack one per trial and time the kernel through six of them.
                gen_adapter.forward, train_adapter.forward = gen_saved, train_saved

            assert gen_seen and train_seen, "adapter was not reached; mega path inactive"
            differed, _ = _routing_disagreements(gen_seen[0], train_seen[0])
            disagreed_total += differed
            worst_output = max(worst_output, _rel_rms(train_out, gen_out))

            close, count, relative = _boundary_gap_stats(train, hidden)
            unresolvable += close
            boundary_total += count
            medians.append(relative.median().item())

        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] boundary at {self.EXPERTS} experts: "
                f"{unresolvable}/{boundary_total} token-boundaries below bf16 resolution "
                f"({100.0 * unresolvable / max(boundary_total, 1):.3f}%), "
                f"median relative gap={sum(medians) / len(medians):.2e}",
                flush=True,
            )
            print(
                f"[mega-metric] routing at {self.EXPERTS} experts: "
                f"{disagreed_total} tokens differ over "
                f"{self.TRIALS * self.TOKENS} sampled",
                flush=True,
            )
        worst_output = _measure_scalar(
            worst_output, f"train/gen output at {self.EXPERTS} experts"
        )

        assert disagreed_total == 0, (
            f"{disagreed_total} of {self.TRIALS * self.TOKENS} tokens routed differently at "
            f"{self.EXPERTS} experts, against 0 at {NUM_EXPERTS}. This is the e2e mechanism "
            "reproduced: the selection boundary is crowded enough that the two paths order "
            "it differently, and one token going to a different expert is worth the 0.26 "
            "nats the run showed."
        )
        assert worst_output < 1e-6, (
            f"routing agrees at {self.EXPERTS} experts but the outputs do not: "
            f"rel_rms={worst_output:.3e}. The divergence is arithmetic, not selection."
        )


def _skew_router(layer, strength, seed):
    """Bias the router per expert so occupancy comes out ragged, not balanced.

    ``_perturb_router`` adds independent noise, which leaves the expert
    *marginals* roughly uniform: every expert still draws about the same number
    of tokens. A shared shift along an expert's whole row moves that expert's
    logit for every token at once, so some experts take most of the traffic and
    others take none -- which is what the real model's routing looks like and
    what decides how the kernel tiles each expert's GEMM.
    """
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if "router" in name and param.dim() == 2:
                generator = torch.Generator(device=param.device).manual_seed(seed)
                bias = torch.randn(
                    (param.shape[0], 1),
                    device=param.device,
                    dtype=param.dtype,
                    generator=generator,
                )
                param.add_(bias * strength)


def _expert_occupancy(indices, num_experts):
    """Tokens per expert for one launch: the quantity the kernel tiles over."""
    counts = torch.bincount(indices.reshape(-1).long(), minlength=num_experts)
    nonzero = counts[counts > 0]
    return {
        "max": int(counts.max().item()),
        "median": int(nonzero.median().item()) if nonzero.numel() else 0,
        "empty": int((counts == 0).sum().item()),
    }


class TestRaggedExpertOccupancy:
    """(9) Per-expert occupancy, which is the axis no test above varies.

    ``test_parity_holds_across_token_counts`` varies the *total* token count
    and finds bitwise equality, so the kernel is not sensitive to launch width
    as such. But it feeds uniform random hidden states, which route evenly:
    2048 tokens over 16 experts leaves every expert near 1024, smooth and
    tail-free on every trial. Total width and per-expert occupancy are
    different variables, and only the first has been tested.

    The real run is nowhere near that. Decode is 8 tokens of top-8 over 128
    experts at EP=8, so most experts receive nothing and the rest receive one
    or two; the scoring pass puts ~23 in each. Both are ragged, neither
    resembles 1024, and the two differ from each other -- which is precisely
    the comparison the e2e KL is made of.

    Ragged occupancy also matches the signature in a way launch width does not.
    A tail that is mishandled at particular occupancies fires for some
    sequences and not others sharing a prompt, hits one token, and lets causal
    attention carry it forward: per-sequence, with an onset, growing after.
    Sensitivity to launch width would instead move every token in the launch,
    which the six bitwise-clean sequences rule out.

    So this compares one wide teacher-forced launch against the same tokens
    replayed at decode width, under router skews that sweep occupancy from
    balanced to concentrated.
    """

    EXPERTS = int(os.environ.get("MEGA_TEST_PROD_EXPERTS", "128"))
    TOKENS = int(os.environ.get("MEGA_TEST_RAGGED_TOKENS", "1024"))
    # The rollout's decode width. Comparing against this is what makes the
    # occupancy on the two sides differ rather than merely be ragged.
    CHUNK = int(os.environ.get("MEGA_TEST_RAGGED_CHUNK", "8"))
    # Balanced, moderately concentrated, and heavily concentrated.
    SKEWS = tuple(
        float(x) for x in os.environ.get("MEGA_TEST_RAGGED_SKEWS", "0,2,6").split(",")
    )

    def _attribute(self, scored, replayed, train_indices, gen_chunks, skew):
        """Report the expert set and both occupancies for each disagreeing token.

        The token's experts are the same on both sides, so what is left to
        report is the only thing that differs: how many tokens shared those
        experts in each launch. ``m_wide`` is the count in the single scoring
        launch, ``m_decode`` the count in the chunk that recomputed the token.
        Those two numbers are the input the kernel tiles from, so they are what
        an upstream report needs.
        """
        try:
            gen_indices = torch.cat(gen_chunks, dim=0)
            differing = (scored - replayed).abs().flatten(1).sum(dim=1).nonzero().flatten()
            if differing.numel() == 0 or train_indices.shape[0] != gen_indices.shape[0]:
                return
            rerouted, _ = _routing_disagreements(
                gen_indices[differing], train_indices[differing]
            )
            if torch.distributed.get_rank() != 0:
                return

            verdict = "selection" if rerouted else "arithmetic (same experts)"
            print(
                f"[mega-metric] attribution at skew={skew}: "
                f"{rerouted}/{differing.numel()} disagreeing tokens rerouted -> {verdict}",
                flush=True,
            )

            wide = torch.bincount(
                train_indices.reshape(-1).long(), minlength=self.EXPERTS
            )
            # Order, not just membership. _routing_disagreements sorts before
            # comparing, on the assumption that the combine is invariant to the
            # order the experts are listed in -- an assumption checked at 16
            # experts and never since. The two paths build the list
            # differently: generation takes InferenceTopKRouter's top-k, while
            # training recovers it from TopKRouter's dense mask via
            # _mega_dense_routing_to_topk.
            train_order = train_indices[differing].reshape(differing.numel(), -1)
            gen_order = gen_indices[differing].reshape(differing.numel(), -1)
            permuted = int((train_order != gen_order).any(dim=-1).sum().item())
            print(
                f"[mega-metric]   of {differing.numel()} disagreeing tokens, "
                f"{permuted} have the same experts in a different order",
                flush=True,
            )

            for token in differing[:3].tolist():
                print(
                    f"[mega-metric]   token {token} order: "
                    f"train={train_indices[token].reshape(-1).long().tolist()} "
                    f"gen={gen_indices[token].reshape(-1).long().tolist()}",
                    flush=True,
                )
                experts = sorted(set(train_indices[token].reshape(-1).long().tolist()))
                chunk = gen_chunks[token // self.CHUNK]
                decode = torch.bincount(chunk.reshape(-1).long(), minlength=self.EXPERTS)
                print(
                    f"[mega-metric]   token {token}: experts={experts} "
                    f"m_wide={[int(wide[e]) for e in experts]} "
                    f"m_decode={[int(decode[e]) for e in experts]}",
                    flush=True,
                )
        except Exception as error:  # diagnostics must not mask the assertion
            if torch.distributed.get_rank() == 0:
                print(f"[mega-metric] attribution unavailable: {error!r}", flush=True)

    def test_generation_matches_scoring_under_ragged_occupancy(self):
        if self.EXPERTS % EP_SIZE:
            pytest.skip(f"{self.EXPERTS} experts do not divide EP={EP_SIZE}")

        shape = dict(
            num_moe_experts=self.EXPERTS,
            inference_mega_max_tokens_per_rank=self.TOKENS,
        )
        gen = _build_layer(_config(mega_training=False, **shape), for_inference=True)
        train = _build_layer(_config(mega_training=True, **shape)).train()

        worst = 0.0
        differed_total = 0
        adapter_worst = 0.0
        adapter_differed = 0
        for index, skew in enumerate(self.SKEWS):
            _skew_router(train, strength=skew, seed=400 + index)
            _copy_expert_weights(train, gen)
            hidden = _hidden(train.config, seed=500 + index, tokens=self.TOKENS)

            adapter = train.experts._mega_training_adapter
            saved = adapter.forward
            seen = []

            def capturing(hidden_states, routing_map, probs, *args, **kwargs):
                seen.append(routing_map.detach().clone())
                return saved(hidden_states, routing_map, probs, *args, **kwargs)

            adapter.forward = capturing
            try:
                with torch.no_grad():
                    scored, _ = train(hidden)
            finally:
                adapter.forward = saved

            # The rollout side: the same tokens, at decode width. Routing is
            # captured per chunk and reassembled so a disagreeing token can be
            # attributed -- a different expert set and the same expert set
            # computed differently are different bugs with different fixes,
            # and the output alone cannot tell them apart.
            gen_adapter = gen.experts._mega_adapter
            gen_saved = gen_adapter.forward
            gen_seen = []

            def gen_capturing(hidden_states, routing_map, probs, *args, **kwargs):
                gen_seen.append(routing_map.detach().clone())
                return gen_saved(hidden_states, routing_map, probs, *args, **kwargs)

            gen_adapter.forward = gen_capturing
            try:
                replayed = _chunked_generation(gen, hidden, chunk=self.CHUNK)
            finally:
                gen_adapter.forward = gen_saved

            # Same width, different module: isolates the adapter instance
            # from the launch width. TestKernelBatchInvarianceByPrecision
            # shows one generation adapter is width-invariant even at this
            # occupancy, so if this arm differs, two adapters is the variable
            # and the kernel's arithmetic is not at fault.
            with torch.no_grad(), InferenceMode.active():
                gen_wide, _ = gen(hidden)
            same_error, same_differed = _measure_token_count_parity(
                scored, gen_wide, f"train vs gen at equal width (skew={skew})"
            )
            adapter_differed += same_differed
            adapter_worst = max(adapter_worst, same_error)

            error, differed = _measure_token_count_parity(
                scored, replayed, f"scoring vs decode-width replay (skew={skew})"
            )
            worst = max(worst, error)
            differed_total += differed

            if differed and seen and gen_seen:
                self._attribute(scored, replayed, seen[0], gen_seen, skew)

            if seen and gen_seen and torch.distributed.get_rank() == 0:
                wide = _expert_occupancy(seen[0], self.EXPERTS)
                # Per chunk, then worst/typical across chunks: the decode side
                # is 32 separate launches, and one aggregate over all of them
                # would describe a launch that never happened.
                per_chunk = [_expert_occupancy(c, self.EXPERTS) for c in gen_seen]
                print(
                    f"[mega-metric] occupancy at skew={skew}: "
                    f"wide(T={self.TOKENS}) max={wide['max']} median={wide['median']} "
                    f"empty={wide['empty']}/{self.EXPERTS} | "
                    f"decode(T={self.CHUNK}) max={max(c['max'] for c in per_chunk)} "
                    f"median={sorted(c['median'] for c in per_chunk)[len(per_chunk) // 2]} "
                    f"empty={min(c['empty'] for c in per_chunk)}-"
                    f"{max(c['empty'] for c in per_chunk)}/{self.EXPERTS}",
                    flush=True,
                )

        if torch.distributed.get_rank() == 0:
            print(
                f"[mega-metric] factored: equal-width train-vs-gen "
                f"{adapter_differed} tokens (rel_rms={adapter_worst:.3e}) | "
                f"wide-vs-decode {differed_total} tokens (rel_rms={worst:.3e}). "
                f"The first column varies only the adapter instance; the second "
                f"also varies launch width.",
                flush=True,
            )

        assert differed_total == 0 and worst < 1e-6, (
            f"{differed_total} tokens differ between one wide scoring launch and the same "
            f"tokens at decode width (rel_rms={worst:.3e}) once expert occupancy is ragged. "
            "This is the e2e comparison in miniature: the rollout computes a token at decode "
            "occupancy and the log-prob pass recomputes it at scoring occupancy, and a token "
            "that disagrees there is one the KL will show."
        )


_MEGA_PRECISIONS = tuple(
    p for p in os.environ.get("MEGA_TEST_PRECISIONS", "bf16,mxfp8").split(",") if p
)


class TestKernelBatchInvarianceByPrecision:
    """(10) Is the occupancy sensitivity a bf16 problem or a kernel problem?

    ``TestRaggedExpertOccupancy`` showed the bf16 megakernel returning
    different bits for the same token, same expert set, when the number of
    tokens sharing those experts changes. That decides whether bf16 can carry
    zero-KL RL; it does not say anything about the quantized precisions, which
    are separate kernels with their own tiling.

    Generation against generation, deliberately. The training forward is
    restricted to bf16 (``moe_mega_training_forward`` rejects everything else,
    since a quantized forward cannot be rebuilt from the parameters after a
    refit), so a train-versus-generation comparison cannot be written for
    mxfp8 at all. Batch invariance does not need one: running the same tokens
    wide and then at decode width exercises the same property, and for bf16 the
    train and generation paths are already known to agree bitwise at equal
    width.

    Only the router is perturbed, never the experts, so the weights FlashInfer
    snapshots and quantizes at build time stay valid across skews.
    """

    EXPERTS = int(os.environ.get("MEGA_TEST_PROD_EXPERTS", "128"))
    TOKENS = int(os.environ.get("MEGA_TEST_RAGGED_TOKENS", "1024"))
    CHUNK = int(os.environ.get("MEGA_TEST_RAGGED_CHUNK", "8"))
    SKEWS = tuple(
        float(x) for x in os.environ.get("MEGA_TEST_RAGGED_SKEWS", "0,2,6").split(",")
    )

    @pytest.mark.parametrize("precision", _MEGA_PRECISIONS)
    def test_generation_is_invariant_to_expert_occupancy(self, precision):
        if self.EXPERTS % EP_SIZE:
            pytest.skip(f"{self.EXPERTS} experts do not divide EP={EP_SIZE}")

        overrides = dict(
            num_moe_experts=self.EXPERTS,
            inference_mega_max_tokens_per_rank=self.TOKENS,
            inference_mega_precision=precision,
        )
        if precision != "bf16":
            # Megatron's fused quantize kernels cover squared-relu only; this
            # spec is SwiGLU, and the mega path quantizes inside FlashInfer
            # regardless.
            overrides["inference_moe_disable_fused_quant_kernels"] = True

        try:
            gen = _build_layer(_config(mega_training=False, **overrides), for_inference=True)
        except (ValueError, NotImplementedError, ImportError) as error:
            pytest.skip(f"{precision} unavailable in this build: {error}")

        worst = 0.0
        differed_total = 0
        for index, skew in enumerate(self.SKEWS):
            _skew_router(gen, strength=skew, seed=700 + index)
            hidden = _hidden(gen.config, seed=800 + index, tokens=self.TOKENS)

            with torch.no_grad(), InferenceMode.active():
                wide, _ = gen(hidden)
            replayed = _chunked_generation(gen, hidden, chunk=self.CHUNK)

            error, differed = _measure_token_count_parity(
                wide, replayed, f"{precision}: wide vs decode-width (skew={skew})"
            )
            worst = max(worst, error)
            differed_total += differed

        if torch.distributed.get_rank() == 0:
            verdict = "NOT batch invariant" if differed_total else "batch invariant"
            print(
                f"[mega-metric] {precision} over {len(self.SKEWS)} skews: "
                f"{differed_total}/{len(self.SKEWS) * self.TOKENS} tokens differ "
                f"-> {verdict}",
                flush=True,
            )

        assert differed_total == 0 and worst < 1e-6, (
            f"the {precision} megakernel is not invariant to expert occupancy: "
            f"{differed_total} of {len(self.SKEWS) * self.TOKENS} tokens differ "
            f"(rel_rms={worst:.3e}) between one wide launch and the same tokens at "
            f"decode width. Zero-KL RL needs the rollout and the log-prob pass to "
            f"agree bitwise, and they run at different widths by construction."
        )

    @pytest.mark.parametrize("precision", _MEGA_PRECISIONS)
    def test_two_instances_agree_under_ragged_occupancy(self, precision):
        """Two separately built layers, identical weights, one width.

        This is the variable the factored ragged run isolated: with launch
        width held equal, the training and generation adapters still disagreed
        on one token, and they differ only in being two FlashInfer instances
        rather than one. mxfp8 has no training forward to compare against, so
        two generation instances stand in -- the question is whether an
        instance resolves its own tile configuration, not which pass owns it.

        Weights are copied before either layer's first forward: the quantized
        precisions let FlashInfer snapshot and quantize during preprocessing,
        and a copy afterwards would not reach the snapshot.
        """
        if self.EXPERTS % EP_SIZE:
            pytest.skip(f"{self.EXPERTS} experts do not divide EP={EP_SIZE}")

        overrides = dict(
            num_moe_experts=self.EXPERTS,
            inference_mega_max_tokens_per_rank=self.TOKENS,
            inference_mega_precision=precision,
        )
        if precision != "bf16":
            overrides["inference_moe_disable_fused_quant_kernels"] = True

        try:
            first = _build_layer(_config(mega_training=False, **overrides), for_inference=True)
            second = _build_layer(_config(mega_training=False, **overrides), for_inference=True)
        except (ValueError, NotImplementedError, ImportError) as error:
            pytest.skip(f"{precision} unavailable in this build: {error}")
        _copy_expert_weights(first, second)

        worst = 0.0
        differed_total = 0
        for index, skew in enumerate(self.SKEWS):
            _skew_router(first, strength=skew, seed=900 + index)
            _copy_expert_weights(first, second)
            hidden = _hidden(first.config, seed=950 + index, tokens=self.TOKENS)

            with torch.no_grad(), InferenceMode.active():
                a, _ = first(hidden)
                b, _ = second(hidden)
            error, differed = _measure_token_count_parity(
                a, b, f"{precision}: two instances, equal width (skew={skew})"
            )
            worst = max(worst, error)
            differed_total += differed

        if torch.distributed.get_rank() == 0:
            verdict = "instance-dependent" if differed_total else "instance-independent"
            print(
                f"[mega-metric] {precision} two instances: {differed_total}/"
                f"{len(self.SKEWS) * self.TOKENS} tokens differ -> {verdict}",
                flush=True,
            )

        assert differed_total == 0 and worst < 1e-6, (
            f"two {precision} instances with identical weights disagree on "
            f"{differed_total} tokens (rel_rms={worst:.3e}) at equal width. Building one "
            "FlashInfer layer per pass is then enough to break bitwise parity on its own, "
            "independent of the kernel's arithmetic."
        )
