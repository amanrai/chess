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
# If no PGNs are provided, this script uses the held-out 2000-2199 eval PGN
# when present, which is the currently available 1800-2200-ish source on the
# target box. If that is missing, it tries to auto-discover files under
# data/processed/lumbras with names containing 1800 and 2200.

if [[ "$#" -gt 0 ]]; then
  PGNS=("$@")
elif [[ -n "${BSE_NEIGHBOR_PGNS:-}" ]]; then
  # shellcheck disable=SC2206
  PGNS=(${BSE_NEIGHBOR_PGNS})
elif [[ -f data/processed/lumbras/eval_board_state_2000_2199/games.pgn ]]; then
  PGNS=(data/processed/lumbras/eval_board_state_2000_2199/games.pgn)
else
  mapfile -t PGNS < <(find data/processed/lumbras -type f -name '*.pgn' \
    | grep -Ei '1800.*2200|2200.*1800|2000.*2199|2199.*2000' \
    | sort)
fi

if [[ "${#PGNS[@]}" -eq 0 ]]; then
  cat >&2 <<'EOF'
No suitable 1800-2200 / 2000-2199 PGN files were provided or auto-discovered.

Run with explicit PGNs, for example:

  ./build_bse_neighbor_index.sh \
    data/processed/lumbras/eval_board_state_2000_2199/games.pgn

Or:

  BSE_NEIGHBOR_PGNS="data/processed/lumbras/file1.pgn data/processed/lumbras/file2.pgn" \
    ./build_bse_neighbor_index.sh
EOF
  exit 2
fi

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
  --device cuda
