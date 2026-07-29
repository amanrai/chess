#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/serve_bse_neighbor_explorer.py \
  --index-dir "${BSE_NEIGHBOR_INDEX_DIR:-data/analysis/bse_neighbors_1800_2200_100k}" \
  --checkpoint "${BSE_CHECKPOINT:-checkpoints/board_state_q_probe_256d_48q/board_state_q_probe_volcanic-planet-22_epoch_001_batch_1350000.pt}" \
  --stockfish-path "${STOCKFISH_PATH:-tools/stockfish/stockfish}" \
  --device "${BSE_NEIGHBOR_DEVICE:-cuda}" \
  --host "${BSE_NEIGHBOR_HOST:-0.0.0.0}" \
  --port "${BSE_NEIGHBOR_PORT:-8765}"
