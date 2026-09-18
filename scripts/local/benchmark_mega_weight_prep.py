#!/usr/bin/env python3
"""Timing breakdown of the mega MoE weight path, bf16 against mxfp8.

Answers one question before a job is submitted: how much wall clock does the
mxfp8 megakernel add, and where. Run it before and after a change to the weight
path to see whether the gap closed.

The thing being measured is not the kernel. It is the repack that hands the
kernel its weights, which the training and log-prob forwards currently redo on
every microbatch. For bf16 that repack is a copy -- interleave gate/up,
transpose to K-major, stack across experts. For mxfp8 it is the same copy plus a
quantization into fp8 with E8M0 block scales, and that quantization is
FlashInfer's own host-side reference implementation (a chain of eager fp32 ops
in ``common/host_utils.py``). We call it deliberately, so our bytes match
FlashInfer's by construction rather than by maintenance -- but FlashInfer calls
it once, when a layer is built, and we call it 48 times per forward.

No distributed setup, no NVSHMEM, no EP group: the repack is local tensor work
on one device, so this runs single-rank in seconds. That is also why it is a
separate script from run_mega_training_tests.sh, which needs the full bootstrap.
The kernel forward itself is deliberately out of scope; it needs that bootstrap
and is measured by the tests.

    ./scripts/local/run_mega_weight_prep_benchmark.sh
    ./scripts/local/run_mega_weight_prep_benchmark.sh --experts 256 --hidden 7168

The projection at the end is the number that matters. Per-forward cost is
multiplied by the layer count and then by the forwards in an RL step, because
the repack is charged per layer per forward and does not scale with tokens --
which is why the log-prob pass, with the fewest tokens to amortize over, was hit
hardest in job 457213.
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

# Presets for the geometries these runs actually use, so the numbers are
# comparable to a job rather than to a unit test's 128-wide toy layer.
MODELS = {
    # examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-*.yaml
    "qwen3-30ba3b": dict(hidden=2048, moe_ffn=768, experts=128, topk=8, layers=48),
    "dsv3": dict(hidden=7168, moe_ffn=2048, experts=256, topk=8, layers=61),
}


def parse_args(argv=None):
    """Command line, defaulting to the model and step shape of job 457213."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=sorted(MODELS), default="qwen3-30ba3b")
    p.add_argument("--hidden", type=int, help="override the preset hidden size")
    p.add_argument("--moe-ffn", type=int, help="override the preset expert FFN size")
    p.add_argument("--experts", type=int, help="override the preset expert count")
    p.add_argument("--layers", type=int, help="override the preset MoE layer count")
    p.add_argument(
        "--ep",
        type=int,
        default=4,
        help="expert-parallel size; only experts/ep land on each rank (default 4)",
    )
    p.add_argument(
        "--precisions",
        default="bf16,mxfp8",
        help="comma-separated; bf16 is the baseline the others are read against",
    )
    p.add_argument("--iters", type=int, default=10, help="timed repacks per precision")
    p.add_argument("--warmup", type=int, default=3, help="untimed repacks first")
    # The defaults reproduce job 457213: train_global_batch_size=256 over DP=4 at
    # train_micro_batch_size=1 is 64 training forwards, and 64 sequences per rank
    # at logprob_batch_size=4 is 16 log-prob forwards.
    p.add_argument("--train-forwards", type=int, default=64)
    p.add_argument("--logprob-forwards", type=int, default=16)
    p.add_argument(
        "--tokens",
        type=int,
        default=8192,
        help="token width to probe the fused activation staging path with",
    )
    args = p.parse_args(argv)
    preset = MODELS[args.model]
    for key in ("hidden", "moe_ffn", "experts", "layers"):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    args.topk = preset["topk"]
    return args


def median_ms(fn, iters, warmup):
    """Median wall clock of ``fn`` in milliseconds, measured on the device.

    Median rather than mean: a single allocator growth or JIT compile on an
    early iteration would otherwise set the number, and those are startup costs
    that a training step does not pay per microbatch.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def expert_parameters(num_local_experts, hidden, moe_ffn, device):
    """Per-expert weights in canonical Megatron layout, as the repack reads them.

    Shapes and dtype are what ``linear_fc1.weight{i}`` / ``linear_fc2.weight{i}``
    hold on a live layer; the values are irrelevant to timing, but they are
    randomized rather than left as ``empty`` so the quantizer's amax reduction
    sees realistic magnitudes instead of denormals or NaNs.
    """
    generator = torch.Generator(device=device).manual_seed(0)
    fc1 = [
        torch.randn(
            2 * moe_ffn, hidden, dtype=torch.bfloat16, device=device, generator=generator
        )
        for _ in range(num_local_experts)
    ]
    fc2 = [
        torch.randn(
            hidden, moe_ffn, dtype=torch.bfloat16, device=device, generator=generator
        )
        for _ in range(num_local_experts)
    ]
    return fc1, fc2


def weight_bytes(num_local_experts, hidden, moe_ffn):
    """Resident bf16 bytes of one layer's local expert weights.

    Reported next to the timings because the repack is bandwidth work on exactly
    this much data, so it sets the floor a fix can reach.
    """
    elements = num_local_experts * (2 * moe_ffn * hidden + hidden * moe_ffn)
    return elements * 2


def time_bf16(num_local_experts, hidden, moe_ffn, device, args):
    """Time the bf16 repack: interleave, transpose, stack. One pass, no compute."""
    from megatron.core.inference.moe.mega.training_weights import MegaKernelWeightBuffer

    fc1, fc2 = expert_parameters(num_local_experts, hidden, moe_ffn, device)
    buffer = MegaKernelWeightBuffer(
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        intermediate_size=moe_ffn,
        dtype=torch.bfloat16,
        device=device,
    )
    total = median_ms(lambda: buffer.repack(fc1, fc2), args.iters, args.warmup)
    return {"stage": total, "total": total}


def time_mxfp8(num_local_experts, hidden, moe_ffn, device, args):
    """Time the mxfp8 repack, split into the three steps it actually performs.

    Timed as separate calls rather than by instrumenting the buffer, so the
    breakdown is the sequence ``MegaMxfp8KernelWeightBuffer.repack`` runs and
    stays honest if that sequence changes: stage into bf16 kernel layout,
    quantize those views, then copy the result into the stable buffers whose
    addresses the kernel (and any CUDA graph) captured.
    """
    from megatron.core.inference.moe.mega.training_weights import (
        MegaKernelWeightBuffer,
        MegaMxfp8KernelWeightBuffer,
        mxfp8_kernel_weights_from_views,
    )

    fc1, fc2 = expert_parameters(num_local_experts, hidden, moe_ffn, device)
    geometry = dict(
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        intermediate_size=moe_ffn,
        dtype=torch.bfloat16,
        device=device,
    )

    staging = MegaKernelWeightBuffer(**geometry)
    stage = median_ms(lambda: staging.repack(fc1, fc2), args.iters, args.warmup)

    views = staging.repack(fc1, fc2)
    quantize = median_ms(
        lambda: mxfp8_kernel_weights_from_views(*views), args.iters, args.warmup
    )

    # The stable-buffer copy, which only happens from the second repack onward:
    # the first fixes the addresses. Driven through the real buffer so the copy
    # measured is the one the production path performs.
    buffer = MegaMxfp8KernelWeightBuffer(cache_staging=True, **geometry)
    buffer.repack(fc1, fc2)
    total = median_ms(lambda: buffer.repack(fc1, fc2), args.iters, args.warmup)
    return {
        "stage": stage,
        "quantize": quantize,
        # By subtraction rather than by timing the copy_ loop directly, which
        # would need the intermediate tensors kept alive and would then measure
        # a different allocator state than the real path sees.
        "copy": max(total - stage - quantize, 0.0),
        "total": total,
    }


def probe_fused_activation_stage(hidden, tokens, device):
    """Whether mxfp8 activation staging takes the fused kernel or the eager path.

    Worth knowing because the fallback is the same eager reference chain as the
    weight quantizer, and it would then run per forward on the generation side
    too. It needs ``hidden % 128`` for mxfp8, so it is a property of the model:
    2048 qualifies, gpt-oss's 2880 does not and silently falls back.
    """
    try:
        from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
            fused_quant_stage_supported,
        )
    except ImportError as error:
        return None, f"probe unavailable: {error}"
    activations = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    supported = fused_quant_stage_supported(activations, quant_type="mxfp8")
    reason = "fused single launch" if supported else "EAGER FALLBACK (hidden % 128 != 0)"
    return supported, reason


def report(results, args, num_local_experts, device):
    """Print the breakdown, then the per-step projection read against bf16."""
    layers = args.layers
    forwards = args.train_forwards + args.logprob_forwards
    resident = weight_bytes(num_local_experts, args.hidden, args.moe_ffn)

    print()
    print(f"geometry      {args.model}: hidden={args.hidden} moe_ffn={args.moe_ffn} "
          f"experts={args.experts} topk={args.topk} layers={layers}")
    print(f"per rank      ep={args.ep} -> {num_local_experts} local experts, "
          f"{resident / 2**30:.2f} GiB of bf16 expert weights per layer")
    print(f"device        {torch.cuda.get_device_name(device)} "
          f"(sm_{torch.cuda.get_device_capability(device)[0]}"
          f"{torch.cuda.get_device_capability(device)[1]})")

    supported, reason = probe_fused_activation_stage(args.hidden, args.tokens, device)
    if supported is not None:
        print(f"activations   mxfp8 staging at {args.tokens} tokens: {reason}")
    else:
        print(f"activations   {reason}")

    print()
    print(f"per-layer repack in ms, median of {args.iters}:")
    print(f"  {'':<10}{'stage':>10}{'quantize':>10}{'copy':>10}{'total':>10}")
    for precision, timings in results.items():
        print(
            f"  {precision:<10}"
            f"{timings.get('stage', 0.0):10.3f}"
            f"{timings.get('quantize', 0.0):10.3f}"
            f"{timings.get('copy', 0.0):10.3f}"
            f"{timings['total']:10.3f}"
        )

    print()
    print(f"projected to a step: {layers} layers x {forwards} forwards "
          f"({args.train_forwards} training + {args.logprob_forwards} log-prob)")
    baseline = results.get("bf16", {}).get("total")
    for precision, timings in results.items():
        per_pass = timings["total"] * layers / 1000.0
        per_step = per_pass * forwards
        line = (
            f"  {precision:<10}{per_pass:7.2f} s per forward pass"
            f"{per_step:9.1f} s per step"
        )
        if baseline is not None and precision != "bf16":
            delta = (timings["total"] - baseline) * layers / 1000.0 * forwards
            line += f"   (+{delta:.1f} s vs bf16)"
        print(line)
    print()
    print("A repack is charged per layer per forward and does not scale with tokens,")
    print("so this is the floor the fix has to remove, not a throughput estimate.")


def main(argv=None):
    """Run the benchmark, or explain what is missing and exit non-zero."""
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA device; this measures device-side repack cost", file=sys.stderr)
        return 2
    if args.experts % args.ep:
        print(f"--experts {args.experts} must be divisible by --ep {args.ep}",
              file=sys.stderr)
        return 2

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    num_local_experts = args.experts // args.ep

    timers = {"bf16": time_bf16, "mxfp8": time_mxfp8}
    results = {}
    for precision in args.precisions.split(","):
        if precision not in timers:
            print(f"no weight packer for {precision!r}; skipping. FlashInfer "
                  "preprocesses and snapshots the others, so there is no repack "
                  "to time -- see mega/QUANTIZED_BLOCKERS.md.", file=sys.stderr)
            continue
        results[precision] = timers[precision](
            num_local_experts, args.hidden, args.moe_ffn, device, args
        )

    if not results:
        print("--precisions selected nothing measurable", file=sys.stderr)
        return 2
    report(results, args, num_local_experts, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
