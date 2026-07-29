#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/build_bse_neighbor_index.py \
  --pgn data/processed/lumbras/<1800_2200_source_1>.pgn data/processed/lumbras/<1800_2200_source_2>.pgn \
  --checkpoint checkpoints/board_state_q_probe_256d_48q/board_state_q_probe_volcanic-planet-22_epoch_001_batch_1350000.pt \
  --out-dir data/analysis/bse_neighbors_1800_2200_100k \
  --nonterminal-samples 90000 \
  --terminal-samples 10000 \
  --min-elo 1800 \
  --max-elo 2200 \
  --bucket-plies 5 \
  --device cuda
