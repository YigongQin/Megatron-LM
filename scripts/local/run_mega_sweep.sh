#!/bin/bash
# Correctness check + benchmark for the flashinfer_mega MoE backend, one run per
# precision, logging each to logs/{check,bench}_<precision>.log.
#
# Must run inside the NeMo RL container: the Megatron worker venv's interpreter
# is not executable from the login node.
#
# Defaults to the DeepSeek-V3 routed-MoE geometry at EP=4 (H=7168, I=2048, 256
# experts, topk=8 -> 64 experts and 5.25 GiB of bf16 expert weights per rank).
#
#   ./scripts/local/run_mega_sweep.sh              # bf16 mxfp8 nvfp4, dsv3, EP=4
#   PRECISIONS=mxfp8 ./scripts/local/run_mega_sweep.sh
#   SHAPE=toy ./scripts/local/run_mega_sweep.sh    # tiny H=I=128 smoke shape
#   GPUS=2 ./scripts/local/run_mega_sweep.sh
#
# MODES selects what each precision runs: check (correctness), bench (one token
# count) and table. MODES=table produces the FlashInfer-style markdown latency
# table over the SWEEP token counts, saved to logs/table_<precision>_<tag>.md
# ready to paste into a PR:
#
#   MODES=table GPUS=8 PRECISIONS=nvfp4 ./scripts/local/run_mega_sweep.sh
#   MODES=table SWEEP=1,8,64,512,4096 ./scripts/local/run_mega_sweep.sh
#
# EXTRA appends flags to whatever SHAPE selected; set it to override a single
# knob (e.g. EXTRA="--local-tokens 512") without restating the geometry.
#
# At --mega-precision mxfp8 the runs automatically include Megatron's own
# torch:mxfp8 grouped GEMM, both as an error yardstick (the "parity" line in
# the check) and as the speed baseline. Everything is measured against
# torch:bf16.
#
# fp8_fp4 is omitted by default because it additionally needs the DeepGEMM
# package; add it to PRECISIONS once deep_gemm imports.
set -u

BASE=/lustre/fsw/portfolios/coreai/users/yigongq/post-training
VENV=$BASE/RL/venvs/infopt-mcore/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
SCRIPT=$BASE/Megatron-LM/scripts/local/benchmark_moe_vllm_vs_mega_bf16.py
LOGDIR=$BASE/Megatron-LM/logs

# This worktree must precede the container's editable Megatron install, and
# fi-main-overlay supplies moe_ep mega kernels absent from flashinfer 0.6.8.
export PYTHONPATH=$BASE/Megatron-LM:$BASE/fi-main-overlay
# Pointing at our own cubin dir sidesteps the flashinfer/flashinfer-cubin
# version check; the mega kernels are CuTeDSL-compiled and use no prebuilt
# cubins, so an empty directory is fine.
export FLASHINFER_CUBIN_DIR=$BASE/fi-main-cubins
export FLASHINFER_WORKSPACE_BASE=$BASE/fi-main-workspace
# The overlay is flashinfer 0.7.0 while the container's flashinfer-jit-cache
# wheel is 0.6.8, and that pairing is checked separately from the cubins above.
# Safe to bypass here: the mega kernels are compiled by CuTeDSL at runtime and
# never read the jit cache, and FLASHINFER_CUBIN_DIR above is empty so no
# stale 0.6.8 artifact can be picked up either.
export FLASHINFER_DISABLE_VERSION_CHECK=1
# CUDA_DEVICE_MAX_CONNECTIONS is deliberately left unset: Blackwell does not
# need it, and TP=1 here regardless.

GPUS=${GPUS:-4}
PRECISIONS=${PRECISIONS:-"bf16 mxfp8 nvfp4"}
SHAPE=${SHAPE:-dsv3}
MODES=${MODES:-"check bench"}
SWEEP=${SWEEP:-flashinfer}
EXTRA=${EXTRA:-}

case "$SHAPE" in
    dsv3) shape_flags="--preset dsv3 --local-tokens 128 --mega-max-tokens 256" ;;
    toy)  shape_flags="" ;;  # script defaults: H=I=128, 8 experts, topk=2
    *)    echo "unknown SHAPE=$SHAPE (expected dsv3 or toy)" >&2; exit 2 ;;
esac

# Logs are tagged by shape so a toy smoke run never overwrites a dsv3 result.
TAG=${TAG:-$SHAPE}
suffix=${TAG:+_$TAG}

mkdir -p "$LOGDIR"
for precision in $PRECISIONS; do
    for mode in $MODES; do
        case "$mode" in
            check) flags="--check" ;;
            bench) flags="--iters 20" ;;
            # 100 iters per row x 14 rows x variants: minutes per precision at
            # dsv3, so it is not part of the default MODES.
            table) flags="--token-sweep $SWEEP --warmup 3 --iters 100" ;;
            *)     echo "unknown mode=$mode (check|bench|table)" >&2; exit 2 ;;
        esac
        log=$LOGDIR/${mode}_${precision}${suffix}.log
        echo "========= $mode $precision ($SHAPE, EP=$GPUS) -> $log ========="
        # shellcheck disable=SC2086  # flag strings are intentionally word-split
        "$VENV/bin/python" -m torch.distributed.run --nproc-per-node="$GPUS" \
            "$SCRIPT" \
            --expert-parallel-size "$GPUS" \
            --mega-precision "$precision" \
            $flags $shape_flags $EXTRA >"$log" 2>&1
        status=$?
        grep -E "\[check rank|\] EP=|time ratio|parity:|correctness:" "$log"
        if [ "$mode" = "table" ]; then
            # Split the markdown out of the log so it can be pasted into a PR
            # description as-is.
            md=$LOGDIR/table_${precision}${suffix}.md
            sed -n '/^Workload:/,$p' "$log" >"$md"
            cat "$md"
        fi
        if [ "$status" -ne 0 ]; then
            # The real exception is near the top; the tail is torchrun killing
            # the surviving ranks, so filter that teardown noise out.
            echo "-- exit=$status; first exception:"
            grep -E "^(\[rank[0-9]+\]: )?[A-Za-z_.]*(Error|Exception):" "$log" \
                | grep -vE "SIGTERM|error_file|ChildFailedError" | head -3
        fi
    done
done
