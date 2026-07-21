DATA_DIR="data/processed/lumbras/verifier"
BOARD_STATE_DIR="data/processed/lumbras/verifier/board_state"
CONTEXT_PLIES="128"
SQUARES_PER_POSITION="16"
BUCKET_PLIES="25"
BATCH_SIZE="32"
GRAD_ACCUM_STEPS="16"
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
CHECKPOINT_DIR="checkpoints/board_state_q_probe"
SNAPSHOT_EVERY_BATCHES="50000"
LOG_WINDOW="1000"
PRINT_EVERY_BATCHES="25"
WANDB_FLAG="--wandb"
WANDB_PROJECT="chess-gm"
WANDB_LOG_EVERY="100"

python scripts/train_board_state_q_probe.py \
  --data-dir "$DATA_DIR" \
  --board-state-dir "$BOARD_STATE_DIR" \
  --context-plies "$CONTEXT_PLIES" \
  --squares-per-position "$SQUARES_PER_POSITION" \
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
  --print-every-batches "$PRINT_EVERY_BATCHES" \
  $WANDB_FLAG \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-log-every "$WANDB_LOG_EVERY"
