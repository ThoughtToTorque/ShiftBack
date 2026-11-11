#!/usr/bin/env bash
# Paired LIBERO evaluation with the paper's settings: one openpi policy server (released pi0.5,
# never changed), every suite run frozen vs ShiftBack on identical initial states and policy noise,
# then the paired statistics.
#
# Usage:
#   OPENPI_DIR=/path/to/openpi LIBERO_ROOT=/path/to/LIBERO sim/run_paired.sh [OUT_ROOT] [SUITE ...]
# Environment (optional): PORT (8000), TRIALS (50 = official protocol), TASK_IDS (0-9), ENV_SEED (7),
#   NOISE_SEED (20260722), EVAL_PYTHON (python of the LIBERO environment), EVAL_ARGS (extra
#   eval_paired.py flags, e.g. ablations).
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
RELEASE=$(dirname "$HERE")
OPENPI_DIR=${OPENPI_DIR:?set OPENPI_DIR to the openpi checkout}
LIBERO_ROOT=${LIBERO_ROOT:?set LIBERO_ROOT to the LIBERO / LIBERO-Pro checkout}
PORT=${PORT:-8000}
TRIALS=${TRIALS:-50}
TASK_IDS=${TASK_IDS:-0-9}
ENV_SEED=${ENV_SEED:-7}
NOISE_SEED=${NOISE_SEED:-20260722}
EVAL_PYTHON=${EVAL_PYTHON:-python}
OUT_ROOT=${1:-$RELEASE/outputs/paired}
shift || true
SUITES=${*:-libero_spatial_swap libero_object_swap libero_goal_swap libero_10_swap libero_spatial libero_object libero_goal libero_10}

# Without these flags XLA autotunes a different gemm/conv per process and the numeric difference
# compounds over a closed-loop rollout, so the same policy on the same episodes flips outcomes
# between runs; with them two runs are bit-identical, which is what makes the paired counts real.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0}"

mkdir -p "$OUT_ROOT"
echo "[server] pi05_libero on :$PORT"
(cd "$OPENPI_DIR" && setsid uv run scripts/serve_policy.py --port "$PORT" \
  policy:checkpoint --policy.config pi05_libero --policy.dir gs://openpi-assets/checkpoints/pi05_libero \
  > "$OUT_ROOT/server.log" 2>&1) &
SERVER_PID=$!
cleanup() { kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 180); do
  if python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(1); s.connect(("127.0.0.1", int(sys.argv[1]))); s.close()' "$PORT" 2>/dev/null; then
    echo "[server] :$PORT up"; break
  fi
  sleep 10
done

cd "$RELEASE"
for suite in $SUITES; do
  run_dir="$OUT_ROOT/$suite"
  if [[ -s "$run_dir/metrics.json" ]]; then
    echo "[resume] $suite already complete"; continue
  fi
  mkdir -p "$run_dir"
  echo "[eval] $suite"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH="$OPENPI_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$EVAL_PYTHON" sim/eval_paired.py --suite "$suite" --task-ids "$TASK_IDS" --trials "$TRIALS" \
    --port "$PORT" --modes frozen,shiftback --libero-root "$LIBERO_ROOT" \
    --seed "$ENV_SEED" --policy-noise-seed "$NOISE_SEED" ${EVAL_ARGS:-} --out "$run_dir" > "$run_dir.log" 2>&1
  "$EVAL_PYTHON" sim/paired_stats.py "$run_dir/rows.json" --plot "$run_dir/per_task.png" | tee "$run_dir/paired.log"
done
echo "=== PAIRED EVAL COMPLETE -> $OUT_ROOT ==="
