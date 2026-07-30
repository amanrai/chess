#!/usr/bin/env bash
set -euo pipefail

# Build the full Lumbras OTB SQLite metadata store.
#
# This streams every raw .7z archive once, writes seekable uncompressed PGN
# shards on /700gpart, and records per-game metadata + byte offsets in SQLite.
#
# Usage:
#   ./build_lumbras_metadata_store.sh
#
# Optional overrides:
#   LUMBRAS_RAW_DIR=data/raw/lumbras/otb
#   LUMBRAS_METADATA_OUT_DIR=/700gpart/chess/data/catalog/lumbras_otb
#   LUMBRAS_METADATA_DB=/custom/path/catalog.sqlite
#   LUMBRAS_METADATA_COMMIT_EVERY=10000
#   LUMBRAS_METADATA_WORKERS=12
#   LUMBRAS_METADATA_FORCE=1

RAW_DIR="${LUMBRAS_RAW_DIR:-data/raw/lumbras/otb}"
OUT_DIR="${LUMBRAS_METADATA_OUT_DIR:-/700gpart/chess/data/catalog/lumbras_otb}"
COMMIT_EVERY="${LUMBRAS_METADATA_COMMIT_EVERY:-10000}"
WORKERS="${LUMBRAS_METADATA_WORKERS:-12}"

ARGS=(
  --raw-dir "$RAW_DIR"
  --out-dir "$OUT_DIR"
  --commit-every "$COMMIT_EVERY"
  --workers "$WORKERS"
)

if [[ -n "${LUMBRAS_METADATA_DB:-}" ]]; then
  ARGS+=(--db "$LUMBRAS_METADATA_DB")
fi

if [[ "${LUMBRAS_METADATA_FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

uv run python scripts/catalog_lumbras_pgn_sqlite.py "${ARGS[@]}" "$@"
