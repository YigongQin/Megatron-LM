#!/bin/bash
# Mega MoE kernel forward, bf16 vs mxfp8 vs nvfp4, swept over live token counts.
#
# Unlike run_mega_weight_prep_benchmark.sh this needs the full EP bootstrap, so
# it runs under torchrun on all GPUs and takes minutes per precision: each one
# bootstraps the NVSHMEM symmetric heap and compiles the kernel through CuTeDSL
# before the first timed forward.
#
#   ./scripts/local/run_mega_kernel_benchmark.sh
#   FI_OVERLAY=0 ./scripts/local/run_mega_kernel_benchmark.sh
#   ./scripts/local/run_mega_kernel_benchmark.sh --model dsv3
#   ./scripts/local/run_mega_kernel_benchmark.sh --precisions bf16,mxfp8
#   ./scripts/local/run_mega_kernel_benchmark.sh --max-tokens-per-rank 40960
#
# Arguments pass through to benchmark_mega_kernel.py; --help lists them. Logs to
# logs/benchmark_mega_kernel.log as well as the terminal, because the per-point
# lines are worth keeping when comparing two runs.
set -u

BASE=/lustre/fsw/portfolios/coreai/users/yigongq/post-training
VENV=$BASE/RL/venvs/infopt-mcore/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"

FI_OVERLAY=${FI_OVERLAY:-1}
if [ "$FI_OVERLAY" = "1" ]; then
    export PYTHONPATH=$REPO:$BASE/fi-main-overlay
else
    export PYTHONPATH=$REPO
fi
export FLASHINFER_CUBIN_DIR=$BASE/fi-main-cubins
export FLASHINFER_WORKSPACE_BASE=$BASE/fi-main-workspace
export FLASHINFER_DISABLE_VERSION_CHECK=1

# GB200/GB300 nodes have 4 GPUs, so 4 ranks is the ceiling; more makes NCCL fail
# with "Multiple Ranks are using the same GPU/Partition". EP is the world size
# because FlashInfer's NVSHMEM bootstrap broadcasts its UID with src=0 as a
# global rank, so an EP group excluding global rank 0 raises there.
GPUS=${GPUS:-4}
# A hung EP collective otherwise burns the full 600 s NCCL watchdog before
# anything is reported. Generous because the CuTeDSL compile is inside it.
TIMEOUT=${TIMEOUT:-1800}

PY=$VENV/bin/python
if [ ! -x "$PY" ]; then
    # The venv's interpreter is a symlink into /root/.local/share/uv (mode 700),
    # because the container builds it as root; -x cannot tell that apart from
    # absent, so both are spelled out.
    if [ -e "$PY" ] || [ -L "$PY" ]; then
        echo "$PY is not executable as $(id -un)." >&2
        echo "  It points at $(readlink "$PY" 2>/dev/null)," >&2
        echo "  which is unreadable unless you are the user that built the venv (root)." >&2
        echo "  Re-run as root inside the container." >&2
    else
        echo "$PY not found. This script must run inside the NeMo RL container." >&2
    fi
    exit 2
fi

free_port() {
    # Let the kernel hand out an unused port. The fixed 29500 default collides
    # with anything else running on the node, which is the most common failure.
    "$PY" - <<'EOF'
import socket

sock = socket.socket()
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
EOF
}

LOG=$LOGDIR/benchmark_mega_kernel.log
port=${MASTER_PORT:-$(free_port)}
echo "========= mega kernel benchmark (ranks=$GPUS, EP=$GPUS) -> $LOG ========="
timeout "$TIMEOUT" "$PY" -m torch.distributed.run \
    --nproc-per-node="$GPUS" --master-port="$port" \
    "$REPO/scripts/local/benchmark_mega_kernel.py" "$@" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "-- exit=$status; full log: $LOG" >&2
fi
exit "$status"
