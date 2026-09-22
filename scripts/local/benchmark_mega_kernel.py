#!/usr/bin/env python3
"""Mega MoE kernel forward, timed per precision at production geometry.

Answers what the megakernel itself costs at bf16 against the quantized
precisions, which is the question left over once the weight repack is accounted
for. Generation is where this matters: a decode step streams the expert weights
and does almost no arithmetic, so mxfp8 at half the weight bytes and nvfp4 at
just over a quarter should be *faster* there, and a measurement that says
otherwise is more likely to be measuring itself than the kernel.

Which is why every point is reported against a memory-bandwidth floor. Decode
is weight-streaming bound -- at qwen3-30ba3b with EP=4 a layer reads 302 MB of
bf16 expert weights, about 38 us at HBM speed -- and a harness that reports
hundreds of microseconds there is dominated by its own per-forward overhead and
cannot see the byte count at all. The floor column is what says whether a number
is worth reading. An earlier version of this script lacked it and concluded
quantized decode was slower, which was an artifact of eager launch overhead.

Each point is therefore timed three ways:

  graph     CUDA graph replay. What generation actually pays, since the
            recipes capture decode. No per-launch submission.
  streamed  Back-to-back forwards, one sync. GPU-resident cost without
            capture.
  isolated  A sync around every forward. Includes submission the GPU waits
            through; the most pessimistic, and the least like production.

Usage, via run_mega_kernel_benchmark.sh which supplies the FlashInfer wiring:

    ./scripts/local/run_mega_kernel_benchmark.sh
    ./scripts/local/run_mega_kernel_benchmark.sh --model dsv3
    ./scripts/local/run_mega_kernel_benchmark.sh --precisions bf16,mxfp8

Expect minutes per precision before the first timed iteration: constructing the
layer bootstraps the NVSHMEM symmetric heap and CuTeDSL compiles the kernel.

Knobs are resolved by FlashInfer from ``max_tokens_per_rank``, not from the live
token count, so ``--max-tokens-per-rank`` is a separate flag and defaults to the
generation recipe's 16384. Comparing across two values of it compares two
different recorded profiles.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Geometries these runs actually use. Shared with benchmark_mega_weight_prep.py
# by convention rather than by import: that script deliberately has no
# distributed dependency, and a shared module would give it one.
MODELS = {
    # examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-*.yaml
    "qwen3-30ba3b": dict(hidden=2048, moe_ffn=768, experts=128, topk=8, layers=48),
    "dsv3": dict(hidden=7168, moe_ffn=2048, experts=256, topk=8, layers=61),
}

# bf16 and mxfp8 build the kernel's weights themselves so a refit can reach
# them; nvfp4 goes through FlashInfer's own preprocessing. The difference
# matters here only because it decides which weights to hand the adapter.
CALLER_OWNED = ("bf16", "mxfp8")

# Stored bytes per weight element, including the block scales, which are the
# whole point of the comparison: mxfp8 is one byte plus one scale byte per 32
# elements, nvfp4 a nibble plus one per 16.
WEIGHT_BYTES = {"bf16": 2.0, "mxfp8": 1 + 1 / 32, "nvfp4": 0.5 + 1 / 16, "fp8_fp4": 0.5 + 1 / 16}

# The three ways each point is timed, in the order they are reported.
MEASUREMENTS = (
    ("graph", "CUDA GRAPH REPLAY (what captured generation pays)"),
    ("streamed", "STREAMED, back-to-back with one sync (GPU-resident cost)"),
    ("isolated", "ISOLATED, a sync per forward (includes launch submission)"),
)


def parse_args(argv=None):
    """Command line, defaulting to the generation side of the mxfp8 recipe."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=sorted(MODELS), default="qwen3-30ba3b")
    p.add_argument("--hidden", type=int, help="override the preset hidden size")
    p.add_argument("--moe-ffn", type=int, help="override the preset expert FFN size")
    p.add_argument("--experts", type=int, help="override the preset expert count")
    p.add_argument("--layers", type=int, help="override the preset MoE layer count")
    p.add_argument("--precisions", default="bf16,mxfp8,nvfp4")
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512, 2048, 8192],
        help="live tokens per rank to sweep: decode widths through prefill",
    )
    p.add_argument(
        "--max-tokens-per-rank",
        type=int,
        default=16384,
        help="the workspace cap, and the key FlashInfer resolves knobs on "
        "(16384 is the generation recipe; training uses 40960)",
    )
    p.add_argument("--iters", type=int, default=20, help="timed forwards per point")
    p.add_argument("--warmup", type=int, default=5, help="untimed forwards first")
    p.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="skip capture; use if it fails and the other two are enough",
    )
    args = p.parse_args(argv)
    preset = MODELS[args.model]
    for key in ("hidden", "moe_ffn", "experts", "layers"):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    args.topk = preset["topk"]
    return args


def build_config(args, precision, ep_size):
    """A TransformerConfig the mega adapter accepts, at the model's geometry.

    Only the fields the megakernel path validates are set. Notably
    ``batch_invariant_mode`` is left off: it replaces attention, the projections
    and the router, none of which this benchmark runs.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=1,
        hidden_size=args.hidden,
        ffn_hidden_size=4 * args.hidden,
        num_attention_heads=8,
        num_query_groups=8,
        num_moe_experts=args.experts,
        moe_ffn_hidden_size=args.moe_ffn,
        moe_router_topk=args.topk,
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
        inference_mega_precision=precision,
        inference_mega_max_tokens_per_rank=args.max_tokens_per_rank,
        expert_model_parallel_size=ep_size,
    )


def build_weights(config, precision, num_local_experts, device):
    """The four weight arguments the adapter takes, for this precision.

    For the caller-owned precisions this is the same buffer production uses, so
    the kernel is reading a layout built the same way. For the rest it is the
    canonical Megatron expert stack, which FlashInfer preprocesses itself.
    """
    hidden = config.hidden_size
    moe_ffn = config.moe_ffn_hidden_size
    generator = torch.Generator(device=device).manual_seed(0)
    # Randomized rather than empty so the quantizers see realistic magnitudes
    # instead of denormals, which can change how fast they run.
    fc1 = [
        torch.randn(2 * moe_ffn, hidden, dtype=torch.bfloat16, device=device, generator=generator)
        for _ in range(num_local_experts)
    ]
    fc2 = [
        torch.randn(hidden, moe_ffn, dtype=torch.bfloat16, device=device, generator=generator)
        for _ in range(num_local_experts)
    ]

    if precision not in CALLER_OWNED:
        # Canonical stacks: [E, 2 * intermediate, hidden] and [E, hidden, intermediate].
        return torch.stack(fc1), torch.stack(fc2), None, None

    from megatron.core.inference.moe.mega.training_weights import (
        MegaKernelWeightBuffer,
        MegaMxfp8KernelWeightBuffer,
    )

    buffer_class = MegaKernelWeightBuffer if precision == "bf16" else MegaMxfp8KernelWeightBuffer
    buffer = buffer_class(
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        intermediate_size=moe_ffn,
        dtype=torch.bfloat16,
        device=device,
    )
    buffer.repack(fc1, fc2)
    views = buffer.views()
    # bf16 has no scale planes; pad so the adapter takes one shape of tuple.
    return views if len(views) == 4 else (*views, None, None)


def make_routing(tokens, num_experts, topk, device, rank):
    """Top-k expert ids and probabilities, in the form the kernel takes.

    Drawn from random scores rather than assigned round-robin, so expert
    occupancy is ragged the way real routing is -- an even split would let every
    expert's GEMM be the same shape and is the easiest way to measure a number
    the kernel never sees in production. Seeded by rank so the ranks do not all
    send their tokens to the same experts.
    """
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    scores = torch.rand(tokens, num_experts, device=device, generator=generator)
    top = scores.topk(topk, dim=-1)
    # fp32 is required by the kernel, and softmax keeps the combine weights in a
    # realistic range rather than uniform.
    return top.indices, torch.softmax(top.values.float(), dim=-1)


def local_experts_touched(ids, num_experts, ep_size, rank):
    """How many of this rank's experts the whole fleet's tokens actually reach.

    The floor a decode step is measured against is set by the weights that get
    streamed, and at small token counts that is well short of every local
    expert: four ranks decoding one token each draw 32 expert slots out of 128,
    so most experts are never read. Counting them makes the floor honest at the
    widths where the answer is in doubt.
    """
    gathered = [torch.empty_like(ids) for _ in range(ep_size)]
    dist.all_gather(gathered, ids.contiguous())
    every = torch.cat([g.reshape(-1) for g in gathered])
    per_rank = num_experts // ep_size
    mine = every[(every >= rank * per_rank) & (every < (rank + 1) * per_rank)]
    return int(torch.unique(mine).numel())


def measure_bandwidth(device, mib=512):
    """Achievable HBM bandwidth in bytes/s, from a large device-to-device copy.

    Measured rather than taken from the spec sheet, so the floor is something
    this machine can actually reach and the ratio against it means something.
    """
    elements = mib * 1024 * 1024 // 2
    source = torch.empty(elements, dtype=torch.bfloat16, device=device)
    destination = torch.empty_like(source)
    for _ in range(3):
        destination.copy_(source)
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(10):
        destination.copy_(source)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1000 / 10
    return 2 * source.numel() * source.element_size() / seconds  # read + write


def _sync_ranks():
    """Line the ranks up, so one starting early does not time the others' skew."""
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()


def _elapsed(body, iters):
    """Milliseconds per iteration of ``body``, run ``iters`` times under one sync."""
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        body()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_forward(adapter, hidden_states, ids, probs, weights, args):
    """Milliseconds per forward, measured the three ways in ``MEASUREMENTS``.

    Graph replay is the one to read for generation: the recipes capture decode,
    so the per-launch submission the other two carry is not paid in production.
    It is attempted last and allowed to fail -- capture has more ways to go
    wrong than the plain paths, and losing it should not cost the whole point.
    """

    def once():
        adapter.forward(hidden_states, ids, probs, *weights)

    for _ in range(args.warmup):
        once()

    _sync_ranks()
    samples = []
    for _ in range(args.iters):
        # Median over individually-synced forwards: the first iterations after a
        # shape change pay allocator growth a steady-state forward does not.
        samples.append(_elapsed(once, 1))
    result = {"isolated": statistics.median(samples)}

    _sync_ranks()
    result["streamed"] = _elapsed(once, args.iters)

    result["graph"] = None
    if not args.no_cuda_graph:
        # Capture needs the work warmed up on a side stream first. The layer is
        # already built and warmed up by now, which capture requires: its
        # NVSHMEM bootstrap and CuTeDSL compile cannot run under capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                once()
        torch.cuda.current_stream().wait_stream(side)
        _sync_ranks()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            once()
        _sync_ranks()
        result["graph"] = _elapsed(graph.replay, args.iters)
    return result


def measure_precision(args, precision, ep_group, ep_size, rank, device, log):
    """Sweep the token counts for one precision, or explain why it could not run.

    Each precision gets its own adapter. They are not shared: the layer binds
    its weights and its workspace at construction, and the geometry key includes
    the precision, so reusing one would either assert or measure the wrong
    kernel.
    """
    from megatron.core.inference.moe.mega.adapter import MegatronMegaMoEAdapter

    config = build_config(args, precision, ep_size)
    num_local_experts = args.experts // ep_size
    weights = build_weights(config, precision, num_local_experts, device)
    adapter = MegatronMegaMoEAdapter(
        config, ep_group, owns_transformed_weights=precision in CALLER_OWNED
    )

    timings = {}
    for tokens in args.tokens:
        if tokens > args.max_tokens_per_rank:
            log(f"  {precision} @ {tokens}: skipped, above the {args.max_tokens_per_rank} cap")
            continue
        hidden_states = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device=device)
        ids, probs = make_routing(tokens, args.experts, args.topk, device, rank)
        # The first forward of the first token count also constructs the layer,
        # bootstraps NVSHMEM and compiles the kernel. Minutes, and one-time.
        point = time_forward(adapter, hidden_states, ids, probs, weights, args)
        point["experts_touched"] = local_experts_touched(ids, args.experts, ep_size, rank)
        timings[tokens] = point
        graph = "n/a" if point["graph"] is None else f"{point['graph']:.3f}"
        log(f"  {precision} @ {tokens} tokens: graph {graph}, "
            f"streamed {point['streamed']:.3f}, isolated {point['isolated']:.3f} ms "
            f"({point['experts_touched']}/{num_local_experts} local experts touched)")
    return timings


def _table(results, precisions, tokens_swept, key, title, log):
    """One timing table: absolute ms, then each precision's ratio against bf16."""
    have = [p for p in precisions if any(results[p][t].get(key) for t in results[p])]
    if not have:
        return
    log(title)
    header = f"  {'tokens':>8}" + "".join(f"{p:>12}" for p in have)
    if "bf16" in have:
        header += "".join(f"{p + ' /bf16':>14}" for p in have if p != "bf16")
    log(header)
    for tokens in tokens_swept:
        if not any(tokens in results[p] for p in have):
            continue
        row = f"  {tokens:>8}"
        for p in have:
            value = results[p].get(tokens, {}).get(key)
            row += f"{value:>12.3f}" if value else f"{'-':>12}"
        baseline = results.get("bf16", {}).get(tokens, {}).get(key)
        for p in have:
            if p == "bf16":
                continue
            value = results[p].get(tokens, {}).get(key)
            row += f"{value / baseline:>13.2f}x" if baseline and value else f"{'-':>14}"
        log(row)
    log("")


def _roofline(results, precisions, args, ep_size, bandwidth, log):
    """Each precision's measured decode cost against the weights it must stream.

    This is the column that decides whether a number is real. A decode forward
    is weight-streaming bound, so the floor is the bytes of expert weight the
    routing reaches divided by achievable bandwidth. A measurement many times
    that floor is reporting overhead, and no change in byte count will move it
    -- which is exactly the trap the first version of this script fell into.
    """
    params = 3 * args.moe_ffn * args.hidden  # fc1 is gate+up, so 2x, plus fc2
    log(f"measured against the weight-streaming floor, at {bandwidth / 1e12:.2f} TB/s:")
    log(f"  {'tokens':>8}{'precision':>11}{'MB read':>10}{'floor ms':>10}"
        f"{'best ms':>10}{'vs floor':>10}")
    for tokens in args.tokens:
        for p in precisions:
            point = results[p].get(tokens)
            if not point:
                continue
            megabytes = point["experts_touched"] * params * WEIGHT_BYTES.get(p, 2.0) / 1e6
            floor = megabytes * 1e6 / bandwidth * 1000
            best = point["graph"] or point["streamed"]
            log(f"  {tokens:>8}{p:>11}{megabytes:>10.1f}{floor:>10.3f}"
                f"{best:>10.3f}{best / floor:>9.1f}x")
    log("")


def report(results, args, ep_size, device, bandwidth, log):
    """The sweep as tables, each precision read against bf16 at the same width."""
    log("")
    log(f"geometry      {args.model}: hidden={args.hidden} moe_ffn={args.moe_ffn} "
        f"experts={args.experts} topk={args.topk}")
    log(f"per rank      ep={ep_size} -> {args.experts // ep_size} local experts")
    log(f"device        {torch.cuda.get_device_name(device)}, "
        f"measured {bandwidth / 1e12:.2f} TB/s device-to-device")
    log(f"workspace     max_tokens_per_rank={args.max_tokens_per_rank} "
        "(also the key FlashInfer resolves knobs on)")
    log("")

    precisions = [p for p in results if results[p]]
    for key, title in MEASUREMENTS:
        _table(
            results, precisions, args.tokens, key,
            f"one MoE layer, ms per forward, {title}:", log,
        )
    _roofline(results, precisions, args, ep_size, bandwidth, log)

    log(f"Per layer. A step runs {args.layers} of these per forward pass, so multiply")
    log("by that and by the forwards per step to reach wall clock. Read the graph table")
    log("for generation. The floor counts weight bytes only, so it bounds the narrow")
    log("rows, where a decode forward is weight-streaming bound and a time far above")
    log("the floor is overhead rather than work. At the wide end arithmetic dominates")
    log("and the floor stops being the binding constraint, so ignore it there.")


def main(argv=None):
    """Run the sweep on every rank; report from rank 0."""
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2

    # torchrun supplies these. Run under run_mega_kernel_benchmark.sh, which
    # picks a free port -- the fixed default collides with anything else on the
    # node, which is how this fails most often.
    if "RANK" not in os.environ:
        print(
            "must run under torchrun: the megakernel owns EP transport and "
            "needs a process group",
            file=sys.stderr,
        )
        return 2
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    ep_size = dist.get_world_size()

    def log(message):
        # Rank 0 only: every rank measures the same collective, and four copies
        # of the table is just noise.
        if rank == 0:
            print(message, flush=True)

    if args.experts % ep_size:
        log(f"--experts {args.experts} must be divisible by the world size {ep_size}")
        return 2

    # The EP group is the whole world. FlashInfer's NVSHMEM bootstrap broadcasts
    # its UID with src=0 as a *global* rank, so an EP group that excludes global
    # rank 0 raises there -- the same constraint run_mega_training_tests.sh
    # defaults EP to the world size for.
    ep_group = dist.group.WORLD
    bandwidth = measure_bandwidth(device)

    results = {}
    for precision in args.precisions.split(","):
        log(f"-- {precision}: building the layer (NVSHMEM bootstrap and a CuTeDSL "
            "compile, minutes on first use)")
        try:
            results[precision] = measure_precision(
                args, precision, ep_group, ep_size, rank, device, log
            )
        except Exception as error:  # pylint: disable=broad-except
            # Reported and skipped rather than fatal, so one precision that has
            # an unmet requirement does not discard the others' measurements.
            # fp8_fp4 needs DeepGEMM, which no worker venv has.
            results[precision] = {}
            log(f"-- {precision}: FAILED, skipping. {type(error).__name__}: {error}")
        # The adapter, its weights and its workspace die with the call, but the
        # caching allocator holds the blocks and the next precision allocates
        # its own set. At dsv3's 64 local experts that is several GB a piece,
        # enough that keeping three rounds of it resident runs the GPU out.
        torch.cuda.empty_cache()

    if any(results.values()):
        report(results, args, ep_size, device, bandwidth, log)
    else:
        log("no precision produced a measurement")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
