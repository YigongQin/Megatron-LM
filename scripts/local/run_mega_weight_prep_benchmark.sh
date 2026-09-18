#!/bin/bash
# Timing breakdown of the mega MoE weight repack, bf16 against mxfp8.
#
# Run this before and after a change to the weight path to see whether the mxfp8
# gap closed. Single rank, no distributed bootstrap, a few seconds.
#
#   ./scripts/local/run_mega_weight_prep_benchmark.sh
#   FI_OVERLAY=0 ./scripts/local/run_mega_weight_prep_benchmark.sh
#   ./scripts/local/run_mega_weight_prep_benchmark.sh --model dsv3
#   ./scripts/local/run_mega_weight_prep_benchmark.sh --ep 8 --iters 20
#
# Arguments are passed through to benchmark_mega_weight_prep.py; --help lists
# them. The FlashInfer wiring below is shared with run_mega_training_tests.sh and
# is the reason this is a script rather than a bare python invocation.
set -u

BASE=/lustre/fsw/portfolios/coreai/users/yigongq/post-training
VENV=$BASE/RL/venvs/infopt-mcore/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
# The tree this script lives in, matching run_mega_training_tests.sh: Megatron-LM
# is developed in place under RL/3rdparty, and that is the copy an RL run
# imports. Override REPO to measure a different checkout.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

# Default to the overlay for the same reason the tests do: the mega kernels are
# absent from the flashinfer wheel the worker venv was originally built against.
# FI_OVERLAY=0 imports the venv's own flashinfer, which is what an RL run uses
# once the venv is rebuilt against the git pin -- the more honest measurement.
FI_OVERLAY=${FI_OVERLAY:-1}
if [ "$FI_OVERLAY" = "1" ]; then
    export PYTHONPATH=$REPO:$BASE/fi-main-overlay
else
    export PYTHONPATH=$REPO
fi
export FLASHINFER_CUBIN_DIR=$BASE/fi-main-cubins
export FLASHINFER_WORKSPACE_BASE=$BASE/fi-main-workspace
export FLASHINFER_DISABLE_VERSION_CHECK=1

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

exec "$PY" "$REPO/scripts/local/benchmark_mega_weight_prep.py" "$@"
