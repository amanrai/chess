#!/usr/bin/env bash
set -euo pipefail

# Serve a small read-only UI for the Lumbras SQLite metadata store.
#
# Usage:
#   ./serve_lumbras_metadata_store.sh
#
# Optional overrides:
#   LUMBRAS_METADATA_DB=/700gpart/chess/data/catalog/lumbras_otb/lumbras_otb_catalog.sqlite
#   LUMBRAS_METADATA_HOST=0.0.0.0
#   LUMBRAS_METADATA_PORT=8770

DB_PATH="${LUMBRAS_METADATA_DB:-/700gpart/chess/data/catalog/lumbras_otb/lumbras_otb_catalog.sqlite}"
HOST="${LUMBRAS_METADATA_HOST:-0.0.0.0}"
PORT="${LUMBRAS_METADATA_PORT:-8770}"

uv run python scripts/serve_lumbras_metadata_store.py \
  --db "$DB_PATH" \
  --host "$HOST" \
  --port "$PORT"
