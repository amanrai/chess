#!/usr/bin/env bash
set -euo pipefail

# Benchmark brute-force top-1 cosine and RMS search over the BSE neighbor index.
#
# By default this runs a smaller sample so we can estimate throughput before
# committing to all 1M queries. Set BSE_SEARCH_QUERIES=0 to query every row once.
#
# Usage:
#   ./benchmark_bse_neighbor_search.sh
#   BSE_SEARCH_QUERIES=0 ./benchmark_bse_neighbor_search.sh

INDEX_DIR="${BSE_NEIGHBOR_INDEX_DIR:-/700gpart/chess/data/analysis/bse_neighbors_1800_2200_1m}"
QUERIES="${BSE_SEARCH_QUERIES:-100}"
CHUNK_SIZE="${BSE_SEARCH_CHUNK_SIZE:-4096}"
DEVICE="${BSE_SEARCH_DEVICE:-cuda}"
SEED="${BSE_SEARCH_SEED:-20260730}"
LOG_EVERY="${BSE_SEARCH_LOG_EVERY:-10}"

ARGS=(
  --index-dir "$INDEX_DIR"
  --queries "$QUERIES"
  --chunk-size "$CHUNK_SIZE"
  --device "$DEVICE"
  --seed "$SEED"
  --log-every "$LOG_EVERY"
)

if [[ "${BSE_SEARCH_EXCLUDE_SAME_GAME:-0}" == "1" ]]; then
  ARGS+=(--exclude-same-game)
fi

if [[ -n "${BSE_SEARCH_OUTPUT:-}" ]]; then
  ARGS+=(--output "$BSE_SEARCH_OUTPUT")
fi

uv run python scripts/benchmark_bse_neighbor_search.py "${ARGS[@]}" "$@"
