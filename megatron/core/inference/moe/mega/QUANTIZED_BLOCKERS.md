# What blocks mxfp8 / nvfp4 / fp8_fp4 on the mega MoE path

`inference_grouped_gemm_backend='flashinfer_mega'` accepts four precisions via
`inference_mega_precision`: `bf16`, `mxfp8`, `nvfp4`, `fp8_fp4`. All four run,
and all four have been benchmarked and correctness-checked as standalone
generation. Only `bf16` is usable for RL train/generation parity today.

This is the list of what stands in the way, separated by which of the two
use cases it blocks, because they are blocked for different reasons and one of
them is a design decision rather than missing work.

## 1. Refit cannot rebuild a quantized kernel weight — blocks RL generation

The hard one. The megakernel does not read Megatron's parameters at forward
time; it reads its own transformed copy of the expert weights. For bf16 that
copy is ours (`MegaKernelWeightBuffer`), which is what makes a refit a repack:
`InferenceGroupedMLP.refresh_mega_weights` marks the buffer stale and the next
forward rewrites it in place, with no teardown, no EP collective and no CuTeDSL
recompile.

The quantized precisions cannot use that buffer. FlashInfer's
`preprocess_mega_weights` does two things — it interleaves gate/up in blocks of
32 and transposes to K-major, *and* it quantizes — and `MegaKernelWeightBuffer`
reproduces only the first. There is no layout we can write bf16 parameters into
that a quantized kernel will read. So those precisions keep
`preprocess_weights=True`, FlashInfer snapshots the weights at construction and
releases the source, and a subsequent refit never reaches the kernel.

`refresh_mega_weights` raises `NotImplementedError` for them rather than
returning quietly. That matters: the failure mode it replaces is silent. Nothing
reads the expert parameters again, so generation would keep sampling from the
weights snapshotted before the refit — finite, plausible rollouts from the
previous step's policy, and the importance ratio that exists to catch
train/generation mismatch would be computed against those same stale log-probs
and would look clean.

Ways out, in increasing order of work:

- **Rebuild the adapter per refit.** Drop `MegatronMegaMoEAdapter._layer` and
  let the next forward reconstruct it. Correct and small, but it pays an EP
  collective, a symmetric-heap reallocation and a CuTeDSL recompile on every
  refit, and construction is illegal under CUDA graph capture, so every refit
  would also invalidate the graphs.
- **Quantize into a caller-owned buffer.** Extend `MegaKernelWeightBuffer` to
  hold the quantized weights and scales and write FlashInfer's quantization
  ourselves. This means reproducing its scale derivation bit-for-bit per
  precision (mxfp8 e4m3 32-element groups, nvfp4 16-element groups with the
  fp8 second-level scale, fp8_fp4's ue8m0 block scales) and keeping it in step
  with FlashInfer. Four formats, each with its own pinning test.
- **Ask FlashInfer for an in-place reload.** A `MoEEpMegaLayer` method that
  re-runs preprocessing into the existing transformed tensors, keeping the
  workspace and the compiled kernel. This is the right place for it — the
  quantizer lives there and stays in step by construction — and it is one
  upstream request rather than four reimplementations.

## 2. A quantized forward against a bf16 backward — blocks the training forward

`moe_mega_training_forward` is rejected for anything but bf16, in
`TransformerConfig.__post_init__`. This is a deliberate refusal, not a gap.

The parity hybrid runs the megakernel for the forward value and takes the
gradient from the MoE recompute pass, which runs the ordinary TE bf16
dispatch + grouped-GEMM path. In bf16 the two differ only by reduction order
and rounding. With a quantized forward they differ by the quantization itself:
the gradient would be of the bf16 function, not of the function that produced
the value. That is a straight-through estimator, which is a training-recipe
decision with convergence consequences — not something a backend flag should
turn on implicitly.

If quantized generation is wanted alongside a bf16 training forward, the honest
framing is that train/generation parity is being given up on purpose, and the
resulting bias belongs in the recipe. Nothing here prevents that combination
from being built; it just should not arrive via `moe_mega_training_forward`.

## 3. Megatron-side MXFP8 is a separate thing, and the two do not compose

Easy to conflate. Two different MXFP8s are in play:

- **Megatron/TE MXFP8** (`--fp8` with `fp8_recipe=mxfp8`), where expert weights
  are stored as `MXFP8Tensor`. Config validation rejects this with
  `flashinfer_mega`: the mega weight packer cannot read an `MXFP8Tensor`.
- **Kernel-side MXFP8** (`inference_mega_precision='mxfp8'`), where parameters
  stay bf16 and FlashInfer quantizes them during preprocessing. This is the one
  the mega path uses.

So a mega MXFP8 run keeps `--fp8` off entirely. Related, and worth knowing
before reaching for TE MXFP8 as an alternative: `InferenceGroupedMLP.forward`
asserts MXFP8 inference-optimized is incompatible with training, so that route
is closed for colocated RL regardless of the megakernel.

## 4. Practicalities, none of them blocking

- **`fp8_fp4` needs DeepGEMM.** It is the only mega kernel not written in
  CuTeDSL; FlashInfer ships the wrapper but does not depend on the package.
  `registry.py` checks `importlib.util.find_spec("deep_gemm")` at config-build
  time so the failure names the package instead of surfacing from inside the
  first forward. Installing it from source needs `libdw-dev` for
  `elfutils/libdwfl.h`, and its `install.sh` calls a bare `python`, which does
  not exist in the container — install with `pip --no-build-isolation` instead.
  It is not installed by default anywhere on this stack: mcore keeps it behind
  its own `batch_invariant` extra, and RL lists it only under `vllm`, so the
  `mcore` worker venv does not have it. Deliberate — nothing but `fp8_fp4` wants
  it, and §1 already rules `fp8_fp4` out for RL, so making every worker venv
  build it from source would buy nothing.
- **Shape alignment is per precision**, enforced in `TransformerConfig` as
  `(hidden_size divisor, moe_ffn_hidden_size divisor)`: bf16 `(32, 64)`,
  mxfp8 `(64, 32)`, nvfp4 `(64, 16)`, fp8_fp4 `(128, 32)`. A geometry that fits
  bf16 need not fit nvfp4.
- **SwiGLU only**, under every precision: the kernel stacks gate+up into w13
  and applies SwiGLU itself, so a non-gated activation has no representable
  weight layout. `activation_func_tanh_clamp_scale` is also rejected — the
  kernel offers a hard FC1 clamp, while that flag is a soft tanh clamp
  replacing the swish gate (SiTU-GLU), and the two are not interchangeable.

## Where the tests are

- `tests/unit_tests/inference/test_mega_training_weights.py` pins our repack
  against FlashInfer's `_interleave_gate_up_32`, and covers the config
  validation above. CPU-only and single rank.
- `tests/unit_tests/inference/test_mega_training_forward.py`,
  `TestGenerationWeightOwnership` covers the bf16 refit path end to end,
  including the refusal in §1. `scripts/local/run_mega_training_tests.sh` with
  `PHASES=gen` runs just that class.
