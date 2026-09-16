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
