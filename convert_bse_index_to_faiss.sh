#!/usr/bin/env bash
set -euo pipefail

# Convert existing BSE neighbor .npy vectors into FAISS indexes.
#
# This does NOT rerun BSE encoding. It reads vectors_raw.fp16.npy and
# vectors_norm.fp16.npy from the existing index directory and writes FAISS files.
#
# Usage:
#   ./convert_bse_index_to_faiss.sh
#
# Optional overrides:
#   BSE_FAISS_INDEX_DIR=/700gpart/chess/data/analysis/bse_neighbors_1800_2200_1m
#   BSE_FAISS_METRIC=both              # both | cosine | l2
#   BSE_FAISS_INDEX_TYPE=ivfpq         # ivfpq | ivfflat | flat
#   BSE_FAISS_NLIST=1024
#   BSE_FAISS_NPROBE=32
#   BSE_FAISS_PQ_M=96
#   BSE_FAISS_PQ_NBITS=8
#   BSE_FAISS_TRAIN_SAMPLES=50000
#   BSE_FAISS_ADD_BATCH_SIZE=4096
#   BSE_FAISS_THREADS=0
#   BSE_FAISS_FORCE=1

ARGS=(
  --index-dir "${BSE_FAISS_INDEX_DIR:-/700gpart/chess/data/analysis/bse_neighbors_1800_2200_1m}"
  --metric "${BSE_FAISS_METRIC:-both}"
  --index-type "${BSE_FAISS_INDEX_TYPE:-ivfpq}"
  --nlist "${BSE_FAISS_NLIST:-1024}"
  --nprobe "${BSE_FAISS_NPROBE:-32}"
  --pq-m "${BSE_FAISS_PQ_M:-96}"
  --pq-nbits "${BSE_FAISS_PQ_NBITS:-8}"
  --train-samples "${BSE_FAISS_TRAIN_SAMPLES:-50000}"
  --add-batch-size "${BSE_FAISS_ADD_BATCH_SIZE:-4096}"
  --sanity-queries "${BSE_FAISS_SANITY_QUERIES:-16}"
  --sanity-k "${BSE_FAISS_SANITY_K:-5}"
  --threads "${BSE_FAISS_THREADS:-0}"
)

if [[ "${BSE_FAISS_FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

uv run python scripts/convert_bse_index_to_faiss.py "${ARGS[@]}" "$@"
