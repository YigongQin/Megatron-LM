#!/bin/bash
# Mamba train/generation parity: does the training SSM forward reproduce the
# kernels generation runs?
#
# Single rank -- unlike the mega MoE suite there is no expert parallelism here,
# and the SSM kernels under test are local. Seconds, not minutes.
#
#   ./scripts/local/run_mamba_parity_tests.sh
#   FI_OVERLAY=0 ./scripts/local/run_mamba_parity_tests.sh
#   PYTEST_ARGS="-k scan" ./scripts/local/run_mamba_parity_tests.sh
#
# MAMBA_DETERMINISTIC=1 is exported here rather than left to the caller because
# the SSM Triton ops fix their autotune config lists inside the
# @triton.autotune decorator, at import. Setting it after the first import of
# megatron.core.ssm changes nothing, and the symptom would be an intermittent
# bitwise failure that reads like a kernel bug. The suite also asserts it, so a
# caller who overrides it gets told rather than a wrong answer.
set -u

BASE=/lustre/fsw/portfolios/coreai/users/yigongq/post-training
VENV=$BASE/RL/venvs/infopt-mcore/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"

export MAMBA_DETERMINISTIC=${MAMBA_DETERMINISTIC:-1}
# Triton >= 3.4 caches autotune results instead of retiming, which is the
# supported way to be deterministic without falling back to the cheapest
# config. determinism.py warns when this is unset.
export TRITON_CACHE_AUTOTUNING=${TRITON_CACHE_AUTOTUNING:-1}

FI_OVERLAY=${FI_OVERLAY:-1}
if [ "$FI_OVERLAY" = "1" ]; then
    export PYTHONPATH=$REPO:$BASE/fi-main-overlay
else
    export PYTHONPATH=$REPO
fi

PY=$VENV/bin/python
if [ ! -x "$PY" ]; then
    # The venv's interpreter is a symlink into /root/.local/share/uv (mode 700)
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

if ! "$PY" -c "import pytest" 2>/dev/null; then
    echo "-- pytest missing from the worker venv; installing it"
    "$PY" -m pip install --quiet pytest || exit 2
fi

free_port() {
    # Let the kernel hand out an unused port rather than taking torchrun's
    # fixed default, which a neighbouring job on a shared node may already
    # hold. Racy in principle, since it is closed before torchrun binds it.
    "$PY" - <<'EOF'
import socket

s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
EOF
}

TEST=$REPO/tests/unit_tests/ssm/test_mamba_train_gen_parity.py
LOG=$LOGDIR/test_mamba_parity.log
eval "pytest_args=( ${PYTEST_ARGS:-} )"

# One rank, but still under torch.distributed.run: the module-level test builds
# a real MambaMixer through Utils.initialize_model_parallel, which needs a
# process group. The kernel-level tests do not care either way, so everything
# runs the same way rather than splitting the suite in two.
port=${MASTER_PORT:-$(free_port)}

echo "========= mamba train/gen parity (MAMBA_DETERMINISTIC=$MAMBA_DETERMINISTIC) -> $LOG ========="
[ ${#pytest_args[@]} -gt 0 ] && echo "-- pytest args: ${pytest_args[*]}"
# -s so the per-pair [mamba-parity] lines reach the log; they are the output
# that matters, including on a pass.
(cd "$REPO" && "$PY" -m torch.distributed.run --nproc-per-node=1 --master-port="$port" \
    -m pytest "$TEST" -q -s -p no:cacheprovider "${pytest_args[@]}") 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo
grep "\[mamba-parity\]" "$LOG" | sort -u
if [ "$status" -ne 0 ]; then
    echo "-- exit=$status; full log: $LOG" >&2
fi
exit "$status"
