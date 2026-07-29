#!/usr/bin/env bash
set -euo pipefail

# Build the 100k BSE nearest-neighbor index.
#
# Usage:
#   ./build_bse_neighbor_index.sh path/to/games_1800_2200.pgn [more.pgn ...]
#
# Or set:
#   BSE_NEIGHBOR_PGNS="path/a.pgn path/b.pgn" ./build_bse_neighbor_index.sh
#
# If no PGNs are provided, this script creates the 1800-2200 Lumbras split
# from the raw .7z archives when needed, then uses that split.

DEFAULT_PGN="data/processed/lumbras/lumbras_otb_both_1800_to_2200_base.pgn"
RAW_DIR="${LUMBRAS_RAW_DIR:-data/raw/lumbras/otb}"
OUT_DIR="${LUMBRAS_PROCESSED_DIR:-data/processed/lumbras}"

if [[ "$#" -gt 0 ]]; then
  PGNS=("$@")
elif [[ -n "${BSE_NEIGHBOR_PGNS:-}" ]]; then
  # shellcheck disable=SC2206
  PGNS=(${BSE_NEIGHBOR_PGNS})
else
  if [[ ! -f "$DEFAULT_PGN" ]]; then
    echo "Creating 1800-2200 Lumbras split at $DEFAULT_PGN"
    uv run python scripts/extract_lumbras_2200_splits.py \
      --raw-dir "$RAW_DIR" \
      --out-dir "$OUT_DIR" \
      --min-elo 1800 \
      --ft-elo 2201
  fi
  PGNS=("$DEFAULT_PGN")
fi

for pgn in "${PGNS[@]}"; do
  if [[ ! -f "$pgn" ]]; then
    echo "PGN not found: $pgn" >&2
    exit 2
  fi
done

printf 'Using PGNs:\n'
printf '  %s\n' "${PGNS[@]}"

uv run python scripts/build_bse_neighbor_index.py \
  --pgn "${PGNS[@]}" \
  --checkpoint checkpoints/board_state_q_probe_256d_48q/board_state_q_probe_volcanic-planet-22_epoch_001_batch_1350000.pt \
  --out-dir data/analysis/bse_neighbors_1800_2200_100k \
  --nonterminal-samples 90000 \
  --terminal-samples 10000 \
  --min-elo 1800 \
  --max-elo 2200 \
  --bucket-plies 5 \
  --max-games "${BSE_NEIGHBOR_MAX_GAMES:-100000}" \
  --device "${BSE_NEIGHBOR_DEVICE:-cuda}"
