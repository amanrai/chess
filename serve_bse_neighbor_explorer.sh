#!/usr/bin/env bash
set -euo pipefail

INDEX_DIR="${BSE_NEIGHBOR_INDEX_DIR:-data/analysis/bse_neighbors_1800_2200_100k}"

if [[ ! -f "$INDEX_DIR/config.json" ]]; then
  mapfile -t CANDIDATES < <(find data/analysis -type f -path '*bse_neighbors*/config.json' 2>/dev/null | sort)
  if [[ "${#CANDIDATES[@]}" -eq 1 ]]; then
    INDEX_DIR="$(dirname "${CANDIDATES[0]}")"
    echo "Using discovered BSE neighbor index: $INDEX_DIR"
  elif [[ "${#CANDIDATES[@]}" -gt 1 ]]; then
    echo "Default index missing: $INDEX_DIR/config.json" >&2
    echo "Multiple BSE neighbor indexes found. Pick one with BSE_NEIGHBOR_INDEX_DIR=..." >&2
    printf '  %s\n' "${CANDIDATES[@]%/config.json}" >&2
    exit 2
  else
    echo "BSE neighbor index not found: $INDEX_DIR/config.json" >&2
    echo "Build it first with:" >&2
    echo "  ./build_bse_neighbor_index.sh" >&2
    exit 2
  fi
fi

uv run python scripts/serve_bse_neighbor_explorer.py \
  --index-dir "$INDEX_DIR" \
  --checkpoint "${BSE_CHECKPOINT:-checkpoints/board_state_q_probe_256d_48q/board_state_q_probe_volcanic-planet-22_epoch_001_batch_1350000.pt}" \
  --stockfish-path "${STOCKFISH_PATH:-tools/stockfish/stockfish}" \
  --device "${BSE_NEIGHBOR_DEVICE:-cuda}" \
  --host "${BSE_NEIGHBOR_HOST:-0.0.0.0}" \
  --port "${BSE_NEIGHBOR_PORT:-8765}"
