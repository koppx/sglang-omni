#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MiniCPM-o 4.5 inference baseline — aligned with issue #2273 methodology.
#
# What this does:
#   1. Launches the MiniCPM-o speech server (single-GPU, DP1, like #2273).
#   2. Sweeps concurrency 1/2/4/8/16/32 on the seed-tts-eval English set.
#   3. Measures speed only (generate-only): req/s, audio s/s, E2E latency.
#   4. Captures a hardware/software fingerprint for reproducibility.
#
# Run it ONCE before touching code (your "before" baseline), then re-run on the
# same card after optimizations and compare the two result tables.
#
# Hardware note:
#   #2273 used 1x H100 SXM 80GB. This script targets A100 80GB. Pick the card
#   with GPU=3 ./run_minicpm_o_baseline.sh  (defaults to GPU 0). Absolute numbers
#   differ from #2273 (H100 is faster); the point is a self-consistent
#   before/after on YOUR card, plus a trend comparison vs #2273.
#
# Env overrides (all optional):
#   MODEL_PATH   HF id or local ckpt dir   (default openbmb/MiniCPM-o-4_5)
#   MODEL_NAME   API model name            (default MiniCPM-o-4_5)
#   PORT         server port               (default 30000)
#   GPU          CUDA_VISIBLE_DEVICES id    (default 0)
#   META         HF dataset id              (default zhaochenyang20/seed-tts-eval-arrow)
#   CONCURRENCY  space-separated sweep     (default "1 2 4 8 16 32")
#   OUT_DIR      results dir                (default results/minicpm_o_baseline_a100)
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-openbmb/MiniCPM-o-4_5}"
MODEL_NAME="${MODEL_NAME:-MiniCPM-o-4_5}"
PORT="${PORT:-30000}"
GPU="${GPU:-0}"
META="${META:-zhaochenyang20/seed-tts-eval-arrow}"  # arrow variant carries the columns this benchmark needs
CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUT_DIR="${OUT_DIR:-results/minicpm_o_baseline_a100}"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4   # #2273 fixed this

mkdir -p "$OUT_DIR"

echo "[baseline] model=$MODEL_PATH  gpu=$GPU  port=$PORT  out=$OUT_DIR"
echo "[baseline] concurrency sweep: $CONCURRENCY"

# --- Capture environment fingerprint (mirrors #2273 baseline fingerprint) --
{
  echo "date:           $(date -Iseconds)"
  echo "git_commit:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "model_path:     $MODEL_PATH"
  echo "cuda_devices:   $CUDA_VISIBLE_DEVICES"
  echo "omp_num_threads: $OMP_NUM_THREADS"
  echo "mem_fraction:   thinker=0.55 talker=0.15"
  echo "max_new_tokens: $MAX_NEW_TOKENS  temperature: $TEMPERATURE"
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
  echo "--- python / torch ---"
  python - <<'PY' 2>/dev/null || true
import torch, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY
} | tee "$OUT_DIR/fingerprint.txt"

# --- 1. Launch server (background) -----------------------------------------
echo "[baseline] launching server on port $PORT ..."
python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" \
  --port "$PORT" \
  --model-name "$MODEL_NAME" \
  --thinker.engine.mem_fraction_static 0.55 \
  --talker.engine.mem_fraction_static 0.15 \
  >"$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "[baseline] stopping server (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 2. Concurrency sweep ---------------------------------------------------
# The benchmark's wait_for_service() blocks until the server is ready, so the
# first run also serves as the readiness gate.
for c in $CONCURRENCY; do
  echo
  echo "[baseline] === concurrency $c ==="
  python -m benchmarks.eval.benchmark_omni_seedtts \
    --meta "$META" \
    --model "$MODEL_NAME" \
    --port "$PORT" \
    --voice-clone \
    --lang en \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --generate-only \
    --max-concurrency "$c" \
    --output-dir "$OUT_DIR/c$c" \
    --disable-tqdm ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}
done

echo
echo "[baseline] done. Per-concurrency results under: $OUT_DIR/c<c>/"
echo "[baseline] server log: $OUT_DIR/server.log   fingerprint: $OUT_DIR/fingerprint.txt"
echo "[baseline] Next: change code, re-run on the SAME GPU, diff the result tables."
