#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MiniCPM-o 4.5 inference baseline — aligned with issue #2273 methodology.
#
# Purpose:
#   A reusable "before/after" measurement harness for the Encoder-cache and
#   Talker optimizations tracked in issue #2284. Run it once before changing
#   code, then re-run on the SAME card after each optimization and diff the
#   per-concurrency run.log files.
#
# What it measures (per concurrency 1/2/4/8/16/32):
#   - throughput (req/s, audio s/s), E2E latency mean/p50/p95, TTFT
#   - GPU memory footprint (idle vs loaded)
#   - optional WER / speaker quality (WITH_Q=1) to prove optimizations did
#     not break audio quality
#
# Hardware:
#   #2273 used 1x H100 SXM 80GB. This targets 1x A100 80GB (pick the card with
#   GPU=3 ...). Absolute numbers differ from #2273; the point is a self-consistent
#   before/after on YOUR card.
#
# Env overrides (all optional):
#   MODEL_PATH   HF id or local ckpt dir   (default openbmb/MiniCPM-o-4_5)
#   MODEL_NAME   API model name            (default MiniCPM-o-4_5)
#   PORT         server port               (default 30000)
#   GPU          CUDA_VISIBLE_DEVICES id    (default 0)
#   META         HF dataset id              (default zhaochenyang20/seed-tts-eval-arrow)
#   CONCURRENCY  space-separated sweep     (default "1 2 4 8 16 32")
#   MAX_SAMPLES  cap samples (smoke test)   (default: full set)
#   WITH_Q       1 = also run WER quality   (default: speed only)
#   OUT_DIR      results dir                (default results/minicpm_o_baseline_a100)
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-openbmb/MiniCPM-o-4_5}"
MODEL_NAME="${MODEL_NAME:-MiniCPM-o-4_5}"
PORT="${PORT:-30000}"
GPU="${GPU:-0}"
META="${META:-zhaochenyang20/seed-tts-eval-arrow}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
WITH_Q="${WITH_Q:-0}"
OUT_DIR="${OUT_DIR:-results/minicpm_o_baseline_a100}"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4   # #2273 fixed this

mkdir -p "$OUT_DIR"

echo "[baseline] model=$MODEL_PATH  gpu=$GPU  port=$PORT  out=$OUT_DIR"
echo "[baseline] concurrency sweep: $CONCURRENCY  with_quality=$WITH_Q"

# --- Environment fingerprint (mirrors #2273 baseline fingerprint) -----------
{
  echo "date:           $(date -Iseconds)"
  echo "git_commit:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "model_path:     $MODEL_PATH"
  echo "cuda_devices:   $CUDA_VISIBLE_DEVICES"
  echo "omp_num_threads: $OMP_NUM_THREADS"
  echo "mem_fraction:   thinker=0.55 talker=0.15"
  echo "max_new_tokens: 256  temperature: 0.7"
  echo "--- nvidia-smi (idle) ---"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
  echo "--- python / torch ---"
  python - <<'PY' 2>/dev/null || true
import torch, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY
} | tee "$OUT_DIR/fingerprint.txt"

# --- 1. Launch server (background) ------------------------------------------
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
# first run doubles as the readiness gate. Each run's stdout is tee'd to
# c<c>/run.log because --generate-only prints the summary to stdout and does
# NOT persist it to disk — we need it for before/after diff.
for c in $CONCURRENCY; do
  echo
  echo "[baseline] === concurrency $c ==="
  mkdir -p "$OUT_DIR/c$c"
  speed_args=(
    --meta "$META"
    --model "$MODEL_NAME"
    --port "$PORT"
    --voice-clone
    --lang en
    --max-new-tokens 256
    --temperature 0.7
    --max-concurrency "$c"
    --output-dir "$OUT_DIR/c$c"
    --disable-tqdm
  )
  [ -n "$MAX_SAMPLES" ] && speed_args+=(--max-samples "$MAX_SAMPLES")

  if [ "$WITH_Q" = "1" ]; then
    # Speed + WER (skips --generate-only so it also runs ASR transcription).
    python -m benchmarks.eval.benchmark_omni_seedtts "${speed_args[@]}" 2>&1 | tee "$OUT_DIR/c$c/run.log"
  else
    # Speed only.
    python -m benchmarks.eval.benchmark_omni_seedtts "${speed_args[@]}" --generate-only 2>&1 | tee "$OUT_DIR/c$c/run.log"
  fi
done

# --- 3. Loaded-memory snapshot ---------------------------------------------
echo
echo "[baseline] --- nvidia-smi (loaded) ---"
nvidia-smi --query-gpu=index,name,memory.used --format=csv 2>/dev/null | tee -a "$OUT_DIR/fingerprint.txt" || true

echo
echo "[baseline] done. Per-concurrency results under: $OUT_DIR/c<c>/run.log"
echo "[baseline] server log: $OUT_DIR/server.log   fingerprint: $OUT_DIR/fingerprint.txt"
echo "[baseline] Before/after: re-run on the SAME GPU after code changes, then"
echo "           diff the c<c>/run.log summary lines (throughput / p50 / TTFT)."
