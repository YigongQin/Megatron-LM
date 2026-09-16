#!/bin/bash
# Tests for the mega MoE training forward (--moe-mega-training-forward), logging
# each phase to logs/test_<phase>.log.
#
# Must run inside the NeMo RL container: the Megatron worker venv's interpreter
# is not executable from the login node. Shares its FlashInfer wiring with
# run_mega_sweep.sh, which is the reason this is a script and not a bare pytest
# invocation — the mega kernels live in the fi-main overlay, not in the
# container's flashinfer wheel.
#
#   ./scripts/local/run_mega_training_tests.sh      # the acceptance plan, ~3 min
#   PHASES=gen ./scripts/local/run_mega_training_tests.sh   # generation only
#   FI_OVERLAY=0 PHASES=gen ./scripts/local/...  # against the venv's flashinfer
#   PHASES=parity ./scripts/local/run_mega_training_tests.sh
#   PHASES=block ./scripts/local/run_mega_training_tests.sh  # whole transformer layer
#   PARITY_RUNS=10 ./scripts/local/run_mega_training_tests.sh
#   PHASES=all ./scripts/local/run_mega_training_tests.sh   # + attribution
#   BI=0 PHASES=parity ./scripts/local/run_mega_training_tests.sh
#   PYTEST_ARGS="-k something" PHASES=forward ./scripts/local/...
#
# The bare invocation runs four phases, in order:
#   weights - layout repack and config validation. CPU-only and single rank, so
#             it passes without Blackwell or the overlay. Seconds.
#   parity  - train/gen forward parity, cold and warm, at one token count and
#             across token counts. Repeated over PARITY_RUNS launches because
#             process launch is the only axis the parity residual has ever
#             varied on: it is bitwise across repeated forwards and across
#             freshly built layer pairs inside one process, so re-running the
#             same launch proves nothing and a second launch proves a lot.
#   loop    - one forward+backward launch: the EP/DP smoke test and the
#             comparison against the standard TE bf16 path. One launch is
#             enough because these assert a tolerance, not bitwise agreement,
#             and bf16 rounding does not vary by launch.
#   gen     - the generation side alone: that the caller-owned weight buffer
#             reproduces FlashInfer's own preprocessing, and that a refit
#             reaches the kernel. Run this before an RL bring-up; a refit that
#             misses the kernel yields plausible rollouts from the previous
#             step's policy rather than an error.
#
# Deliberately not in the default plan is the attribution phase, four tests
# written to localize the parity residual by holding one variable at a time
# (the weight path, the construction, the weight update). Ordering and
# construction are both settled now, so they are diagnostics to reach for when
# parity regresses rather than an acceptance gate: PHASES=attribution.
#
# GPUS is the rank count and EP the expert-parallel size, so GPUS/EP would be
# the data-parallel replication of each expert shard. EP defaults to GPUS,
# i.e. no replication, because FlashInfer cannot currently bootstrap an EP
# group that excludes global rank 0 (see the EP default below); the
# expert-data-parallel agreement test skips itself in that case.
set -u

BASE=/lustre/fsw/portfolios/coreai/users/yigongq/post-training
VENV=$BASE/RL/venvs/infopt-mcore/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
# The tree this script lives in, rather than a fixed path. Megatron-LM is
# developed in place under RL/3rdparty, which is where the editable
# megatron-core install resolves to and therefore the only copy an RL run
# imports; testing the standalone worktree instead would measure bytes no run
# will execute. Override REPO to test a different checkout.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
LOGDIR=$REPO/logs

# This tree must precede the container's editable Megatron install. The overlay
# then supplies the moe_ep mega kernels, which are absent from the flashinfer
# 0.6.8 wheel the worker venv was originally built against.
#
# Note it shadows the venv's own flashinfer, being earlier on PYTHONPATH. Once
# the venv is rebuilt against the 0.7.0 git pin in RL's pyproject.toml, moe_ep
# is in the venv and FI_OVERLAY=0 is the more honest test: it imports the same
# flashinfer an RL run will.
FI_OVERLAY=${FI_OVERLAY:-1}
if [ "$FI_OVERLAY" = "1" ]; then
    export PYTHONPATH=$REPO:$BASE/fi-main-overlay
else
    export PYTHONPATH=$REPO
fi
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
# need it, and TP=1 in every case here.

# GB200 nodes have 4 GPUs, so 4 ranks is the ceiling here; asking for more makes
# NCCL fail with "Multiple Ranks are using the same GPU/Partition".
GPUS=${GPUS:-4}
# EP defaults to the full world because flashinfer's NVSHMEM bootstrap
# broadcasts its UID with dist.broadcast(src=0, group=ep_group), and src is a
# global rank: every EP group that does not contain global rank 0 raises there
# while the group that does blocks in the following collective, hanging the run.
# EP<GPUS therefore needs Megatron to bring NVSHMEM up over the EP group itself,
# which is not wired in yet; the replica-agreement test skips until it is.
EP=${EP:-$GPUS}
# A desynchronized EP collective otherwise burns the full 600 s NCCL watchdog
# timeout before anything is reported.
TIMEOUT=${TIMEOUT:-420}
PHASES=${PHASES:-"weights parity loop gen block"}
# The plan plus the diagnostics, which together are every test in both files.
if [ "$PHASES" = "all" ]; then
    PHASES="weights parity loop gen block attribution"
fi
# Three launches keeps the default under the time budget while still being
# informative: the residual ran ~6/10 launches before batch-invariant mode, so
# if nothing had changed, three clean launches would be a 0.4^3 ~ 6% fluke.
# Raise it when a regression needs pinning down.
PARITY_RUNS=${PARITY_RUNS:-3}
ATTRIBUTION_RUNS=${ATTRIBUTION_RUNS:-1}
# Overrides the phase's own -k selector. Only useful with PHASES=forward, which
# is the whole file and selects nothing by default.
PYTEST_ARGS=${PYTEST_ARGS:-}
# The worker venv ships no pytest (it only ever runs plain scripts), so bootstrap
# it on first use. Set INSTALL_PYTEST=0 to fail instead of installing.
INSTALL_PYTEST=${INSTALL_PYTEST:-1}

PY=$VENV/bin/python
FORWARD=tests/unit_tests/inference/test_mega_training_forward.py

export MEGA_TEST_EP_SIZE=$EP
# On by default because it is the configuration that is actually bitwise, and
# the one RL will run: batch-invariant mode replaces attention, the projections
# and the router, mega replaces only the expert compute, and they cover disjoint
# parts of the layer. The router is what matters here — it is the last place the
# two paths ran different code, since InferenceTopKRouter uses a torch.compile'd
# top-k unless batch-invariant mode swaps in the same eager function training
# calls. With it off, a fixed 1/128 of tokens disagree at ~6e-5; with it on the
# forward is bitwise. BI=0 measures the difference.
export MEGA_TEST_BATCH_INVARIANT=${BI:-1}
# Three is enough to catch a forward that is not reproducible within a process.
# Repeats were originally 20 to hunt the parity residual, which turned out not
# to vary on this axis at all, so the other 17 were only costing time.
export MEGA_TEST_PARITY_REPEATS=${MEGA_TEST_PARITY_REPEATS:-3}

if [ $((GPUS % EP)) -ne 0 ]; then
    echo "GPUS=$GPUS must be a multiple of EP=$EP" >&2
    exit 2
fi

if [ ! -x "$PY" ]; then
    # Two different failures, and -x cannot tell them apart: the venv's
    # interpreter is a symlink into /root/.local/share/uv (mode 700), because
    # the container builds the venv as root. A non-root shell sees the symlink
    # but cannot stat its target, which -x reports the same as absent.
    if [ -e "$PY" ] || [ -L "$PY" ]; then
        echo "$PY is not executable as $(id -un)." >&2
        echo "  It points at $(readlink "$PY" 2>/dev/null)," >&2
        echo "  which is unreadable unless you are the user that built the venv (root)." >&2
        echo "  Re-run as root inside the container, or rebuild the venv with" >&2
        echo "  UV_PYTHON_INSTALL_DIR set outside /root." >&2
    else
        echo "$PY not found. This script must run inside the NeMo RL container." >&2
    fi
    exit 2
fi

if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
    if [ "$INSTALL_PYTEST" != "1" ]; then
        echo "pytest is missing from $VENV and INSTALL_PYTEST=0." >&2
        echo "Install it with: $PY -m pip install pytest" >&2
        exit 3
    fi
    echo "-- pytest missing from the worker venv; installing it"
    if ! "$PY" -m pip install --quiet pytest; then
        echo "pytest install failed (no network in the container?)." >&2
        echo "Install it manually, then re-run: $PY -m pip install pytest" >&2
        exit 3
    fi
fi

mkdir -p "$LOGDIR"
status_all=0
started=$SECONDS
echo "========= plan: $PHASES (ranks=$GPUS, EP=$EP, BI=$MEGA_TEST_BATCH_INVARIANT) ========="
for phase in $PHASES; do
    # Each phase is a target file, a -k selector, and a launch count. Splitting
    # the forward file this way is what keeps the default plan cheap: the whole
    # file is 11 tests, and repeating all of them across launches would spend
    # most of the budget re-measuring things that do not vary by launch.
    case "$phase" in
        # No model, no collective, no CUDA: run it directly. Going through
        # torchrun would only wrap any failure in a ChildFailedError that hides
        # the actual assertion.
        weights)
            ranks=1
            target=tests/unit_tests/inference/test_mega_training_weights.py
            selector=""
            launches=1
            ;;
        parity)
            ranks=$GPUS
            target=$FORWARD
            # token_count is the strictest of the three and subsumes a plain
            # cold-parity check: it builds both layers fresh, so its first
            # forward is the cold one, and it also varies the token count.
            # generation_forward is kept for its explicit cold/warm split, and
            # batch_invariant is the gen-vs-gen control that says whether a
            # token-count failure is the kernel's or ours.
            selector="-k 'generation_forward or token_count or batch_invariant'"
            launches=$PARITY_RUNS
            ;;
        loop)
            ranks=$GPUS
            target=$FORWARD
            selector="-k 'ep_and_dp or replicas_agree or te_bf16'"
            launches=1
            ;;
        # The generation side on its own: the caller-owned weight buffer and the
        # refit hook. Worth running alone before an RL bring-up, since a refit
        # that does not reach the kernel produces plausible rollouts from the
        # previous step's policy rather than an error.
        gen)
            ranks=$GPUS
            target=$FORWARD
            selector="-k 'TestGenerationWeightOwnership'"
            launches=1
            ;;
        # The MoE layer inside a whole transformer layer, so attention and the
        # projections are in the picture. Enabling mega on the training side
        # also moves that surrounding code onto the inference-optimized spec,
        # which every other phase here builds around rather than through.
        block)
            ranks=$GPUS
            target=$FORWARD
            selector="-k 'TestWholeTransformerLayer'"
            launches=1
            ;;
        attribution)
            ranks=$GPUS
            target=$FORWARD
            selector="-k 'identically_built or layer_constructions or weight_update'"
            launches=$ATTRIBUTION_RUNS
            ;;
        # Escape hatch: the whole forward file, or a hand-written PYTEST_ARGS.
        forward)
            ranks=$GPUS
            target=$FORWARD
            selector=""
            launches=1
            ;;
        *)
            echo "unknown phase=$phase (weights|parity|loop|gen|block|attribution|forward|all)" >&2
            exit 2
            ;;
    esac
    # A hand-written selector replaces the phase's own, so PHASES and
    # PYTEST_ARGS cannot silently intersect to nothing.
    if [ -n "$PYTEST_ARGS" ]; then
        selector=$PYTEST_ARGS
    fi
    if [ "$ranks" -eq 1 ]; then
        launcher=""
    else
        launcher="-m torch.distributed.run --nproc-per-node=$ranks"
    fi
    # Stale per-launch logs would otherwise be folded into the tally below.
    rm -f "$LOGDIR"/test_"${phase}".run*.log
    phase_started=$SECONDS
    for run in $(seq 1 "$launches"); do
        if [ "$launches" -eq 1 ]; then
            log=$LOGDIR/test_${phase}.log
            echo "========= $phase (ranks=$ranks) -> $log ========="
        else
            log=$LOGDIR/test_${phase}.run${run}.log
            echo "========= $phase run $run/$launches (ranks=$ranks) -> $log ========="
        fi
        # Echo the selection flags: an inherited PYTEST_ADDOPTS silently
        # deselects tests, which reads as a pass rather than a skip.
        if [ -n "$selector" ] || [ -n "${PYTEST_ADDOPTS:-}" ]; then
            echo "-- pytest args: $selector ${PYTEST_ADDOPTS:-}"
        fi
        # Re-parse the selector the way the shell would, so a quoted -k 'a or b'
        # stays one argument. Expanding it unquoted instead splits on the spaces
        # inside the quotes and pytest exits 4 on "not found: or".
        eval "pytest_args=( ${selector} )"
        # -p no:cacheprovider keeps concurrent ranks from fighting over .pytest_cache.
        # -s so the [mega-metric] lines survive: pytest discards captured stdout for
        # passing tests, which is exactly when the measured margin is interesting.
        # shellcheck disable=SC2086  # launcher is intentionally split
        (cd "$REPO" && timeout --signal=INT "$TIMEOUT" "$PY" $launcher \
            -m pytest -q -s -p no:cacheprovider --no-header \
            "$target" ${pytest_args[@]+"${pytest_args[@]}"}) >"$log" 2>&1
        status=$?
        if [ "$status" -eq 124 ] || [ "$status" -eq 130 ]; then
            echo "-- TIMED OUT after ${TIMEOUT}s: ranks are almost certainly split" \
                 "across a collective. Check which rank is missing from the last" \
                 "barrier in $log."
        fi

        # pytest's own summary, then the measured rel_rms for every comparison,
        # reported whether it passed or failed so the margin is always visible.
        grep -E "[0-9]+ (passed|failed|skipped|error)" "$log" | tail -3
        grep -E "^\[mega-metric\] " "$log" | sort -u
        grep -E "parity broken|diverges|received no gradient" "$log" | sort -u
        if [ "$status" -ne 0 ]; then
            status_all=$status
            echo "-- exit=$status; cause:"
            # torchrun buries the real failure: everything from "Traceback (most
            # recent call last)" onward is it re-raising ChildFailedError in the
            # launcher, so report what the rank itself printed before that. Falling
            # back to the head of the log matters for setup failures (a missing
            # module, an import error) that never reach a pytest assertion.
            cause=$(grep -E "^E |assert|ModuleNotFoundError|ImportError|No module named|[A-Za-z_.]+(Error|Exception):" "$log" \
                | grep -vE "SIGTERM|error_file|ChildFailedError|elastic|launch_agent" | head -5)
            if [ -n "$cause" ]; then
                echo "$cause"
            else
                sed -n '/Traceback (most recent call last)/q;p' "$log" | tail -12
            fi
            echo "-- full log: $log"
        fi
    done

    # Every metric is printed as %.3e, so bitwise agreement is exactly the string
    # 0.000e+00 and anything else is a real difference. Counting launches rather
    # than lines is the point: the residual has only ever shown up between
    # launches, never between forwards or builds inside one.
    #
    # The labels are tallied apart because together they attribute the cause.
    # "train/gen parity" crosses the two weight paths (FlashInfer's own
    # preprocessing vs the caller-owned repack); "two identically built layers"
    # holds the weight path fixed. Both disagreeing means the kernel is not
    # reproducible across instances; only the first disagreeing means the repack
    # differs from FlashInfer's transform.
    if [ "$launches" -gt 1 ]; then
        for label in "train/gen parity" "two identically built layers" \
                     "two identically built gen layers" \
                     "token-count parity (train vs gen)" \
                     "token-count parity (gen vs gen)"; do
            disagreed=0
            seen=0
            for f in "$LOGDIR"/test_"${phase}".run*.log; do
                metrics=$(grep -F "[mega-metric] $label" "$f")
                [ -n "$metrics" ] || continue
                seen=$((seen + 1))
                if echo "$metrics" | grep -oE "[0-9]\.[0-9]{3}e[+-][0-9]{2}" \
                     | grep -qv '^0\.000e+00$'; then
                    disagreed=$((disagreed + 1))
                    echo "-- $(basename "$f") disagreed on '$label':"
                    echo "$metrics" | sed 's/^/     /'
                fi
            done
            if [ "$seen" -gt 0 ]; then
                echo "========= $phase: '$label' disagreed in" \
                     "$disagreed/$seen launches ========="
            fi
        done
    fi
    # Reported so the plan can be retuned against real numbers rather than
    # guesses: PARITY_RUNS is the knob, and this is its unit cost.
    echo "========= $phase: $((SECONDS - phase_started))s over $launches launch(es) ========="
done

echo "========= total: $((SECONDS - started))s ========="
exit "$status_all"
