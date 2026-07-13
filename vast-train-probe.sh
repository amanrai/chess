#!/usr/bin/env bash
# Train the Q-Former fact probe on Vast.
#
# Override any setting with an environment variable, for example:
#   BATCH_SIZE=64 EPOCHS=2 WANDB=1 bash vast-train-probe.sh
# Extra CLI arguments are appended last and therefore override script defaults:
#   bash vast-train-probe.sh --batch-size 64 --epochs 2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/processed/lumbras/verifier}"
CONTEXT_PLIES="${CONTEXT_PLIES:-125}"
MAX_PROBE_PLIES="${MAX_PROBE_PLIES:-250}"
BUCKET_PLIES="${BUCKET_PLIES:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MODEL_DIM="${MODEL_DIM:-256}"
HEADS="${HEADS:-16}"
HISTORY_LAYERS="${HISTORY_LAYERS:-4}"
Q_LAYERS="${Q_LAYERS:-2}"
NUM_QUERIES="${NUM_QUERIES:-16}"
DROPOUT="${DROPOUT:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/q_probe}"
SNAPSHOT_EVERY_BATCHES="${SNAPSHOT_EVERY_BATCHES:-5000}"
LOG_WINDOW="${LOG_WINDOW:-1000}"
CHECK_POSITIVE_WEIGHT="${CHECK_POSITIVE_WEIGHT:-20.0}"
CHECK_POSITIVE_WEIGHT_END="${CHECK_POSITIVE_WEIGHT_END:-1.0}"
CHECK_POSITIVE_WEIGHT_DECAY_BATCHES="${CHECK_POSITIVE_WEIGHT_DECAY_BATCHES:-150000}"
MATE_POSITIVE_WEIGHT="${MATE_POSITIVE_WEIGHT:-50.0}"
WANDB="${WANDB:-auto}" # auto, 1, or 0
WANDB_PROJECT="${WANDB_PROJECT:-chess-gm}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-100}"

args=(
  --data-dir "$DATA_DIR"
  --context-plies "$CONTEXT_PLIES"
  --max-probe-plies "$MAX_PROBE_PLIES"
  --bucket-plies "$BUCKET_PLIES"
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --model-dim "$MODEL_DIM"
  --heads "$HEADS"
  --history-layers "$HISTORY_LAYERS"
  --q-layers "$Q_LAYERS"
  --num-queries "$NUM_QUERIES"
  --dropout "$DROPOUT"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --snapshot-every-batches "$SNAPSHOT_EVERY_BATCHES"
  --log-window "$LOG_WINDOW"
  --check-positive-weight "$CHECK_POSITIVE_WEIGHT"
  --check-positive-weight-end "$CHECK_POSITIVE_WEIGHT_END"
  --check-positive-weight-decay-batches "$CHECK_POSITIVE_WEIGHT_DECAY_BATCHES"
  --mate-positive-weight "$MATE_POSITIVE_WEIGHT"
  --wandb-project "$WANDB_PROJECT"
  --wandb-log-every "$WANDB_LOG_EVERY"
)

# Optional probe-trainer inputs. Leave unset to preserve its native defaults.
[[ -n "${MIN_GAME_PLIES:-}" ]] && args+=(--min-game-plies "$MIN_GAME_PLIES")
[[ -n "${MAX_GAME_PLIES:-}" ]] && args+=(--max-game-plies "$MAX_GAME_PLIES")
[[ -n "${EXAMPLES_PER_EPOCH:-}" ]] && args+=(--examples-per-epoch "$EXAMPLES_PER_EPOCH")
[[ -n "${INIT_CHECKPOINT:-}" ]] && args+=(--init-checkpoint "$INIT_CHECKPOINT")
[[ -n "${WANDB_RUN_NAME:-}" ]] && args+=(--wandb-run-name "$WANDB_RUN_NAME")

case "$WANDB" in
  1|true|yes) args+=(--wandb) ;;
  0|false|no) ;;
  auto)
    if [[ -n "${WANDB_API_KEY:-}" ]] || grep -qE '^wandb_key=' .env 2>/dev/null; then
      args+=(--wandb)
    fi
    ;;
  *) echo "invalid WANDB=$WANDB (expected auto, 1, or 0)" >&2; exit 2 ;;
esac

exec "$PYTHON_BIN" scripts/train_encoder_q_probe.py "${args[@]}" "$@"
