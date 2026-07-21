DATA_DIR="data/processed/lumbras/verifier"
CONTEXT_PLIES="128"
MAX_PROBE_PLIES="250"
BUCKET_PLIES="25"
BATCH_SIZE="32"
GRAD_ACCUM_STEPS="1"
EPOCHS="1"
LEARNING_RATE="3e-4"
WEIGHT_DECAY="0.01"
ENCODER_INTERNAL_DIM="512"
ENCODER_ATTENTION_HEADS="32"
ENCODER_HISTORY_LAYERS="6"
Q_PROBE_LAYERS="4"
Q_PROBE_QUERY_SLOTS="64"
DROPOUT="0.0"
DATALOADER_WORKERS="4"
DEVICE="cuda"
CHECKPOINT_DIR="checkpoints/q_probe"
SNAPSHOT_EVERY_BATCHES="50000"
LOG_WINDOW="1000"
CHECK_POSITIVE_WEIGHT="5.0"
CHECK_POSITIVE_WEIGHT_END="1.0"
CHECK_POSITIVE_WEIGHT_DECAY_BATCHES="150000"
MATE_POSITIVE_WEIGHT="50.0"
WANDB_FLAG="--wandb"
WANDB_PROJECT="chess-gm"
WANDB_LOG_EVERY="100"

uv run python scripts/train_encoder_q_probe.py \
  --data-dir "$DATA_DIR" \
  --context-plies "$CONTEXT_PLIES" \
  --max-probe-plies "$MAX_PROBE_PLIES" \
  --bucket-plies "$BUCKET_PLIES" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --model-dim "$ENCODER_INTERNAL_DIM" \
  --heads "$ENCODER_ATTENTION_HEADS" \
  --history-layers "$ENCODER_HISTORY_LAYERS" \
  --q-layers "$Q_PROBE_LAYERS" \
  --num-queries "$Q_PROBE_QUERY_SLOTS" \
  --dropout "$DROPOUT" \
  --num-workers "$DATALOADER_WORKERS" \
  --device "$DEVICE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --snapshot-every-batches "$SNAPSHOT_EVERY_BATCHES" \
  --log-window "$LOG_WINDOW" \
  --check-positive-weight "$CHECK_POSITIVE_WEIGHT" \
  --check-positive-weight-end "$CHECK_POSITIVE_WEIGHT_END" \
  --check-positive-weight-decay-batches "$CHECK_POSITIVE_WEIGHT_DECAY_BATCHES" \
  --mate-positive-weight "$MATE_POSITIVE_WEIGHT" \
  $WANDB_FLAG \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-log-every "$WANDB_LOG_EVERY"
