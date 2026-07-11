#!/usr/bin/env bash
# One-shot Vast setup: install deps, download Lumbras data, preprocess verifier store, optionally train.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKERS="${WORKERS:-$(nproc)}"
CHUNKSIZE="${CHUNKSIZE:-128}"
RUN_TRAIN="${RUN_TRAIN:-probe}"
SKIP_SYNC="${SKIP_SYNC:-0}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/vast_download_preprocess_run.sh [options]

Options:
  --run {probe|qverifier|none}  What to run after preprocessing. Default: probe
  --workers N                  Preprocess workers. Default: nproc
  --chunksize N                Preprocess chunksize. Default: 128
  --skip-apt                   Do not apt-install system packages
  --skip-sync                  Do not run uv sync
  --skip-download              Do not download Lumbras archives
  --skip-extract               Do not extract/split Lumbras PGNs
  --skip-preprocess            Do not build verifier game store
  -h, --help                   Show this help

Environment overrides:
  RUN_TRAIN=probe|qverifier|none
  WORKERS=N
  CHUNKSIZE=N
  SKIP_APT=1 SKIP_SYNC=1 SKIP_DOWNLOAD=1 SKIP_EXTRACT=1 SKIP_PREPROCESS=1

Default probe command after preprocessing:
  uv run python scripts/train_encoder_q_probe.py --batch-size 32 --context-plies 125 --max-probe-plies 250 --bucket-plies 25 --model-dim 256 --heads 16 --grad-accum-steps 16

Default q-verifier command after preprocessing:
  uv run python scripts/train_encoder_q.py --batch-size 32 --sample-mode prefix --prefix-fraction 0.5 --context-plies 125 --min-game-plies 100 --max-game-plies 250 --model-dim 256 --heads 16 --grad-accum-steps 16
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_TRAIN="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --chunksize) CHUNKSIZE="$2"; shift 2 ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --skip-sync) SKIP_SYNC=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-extract) SKIP_EXTRACT=1; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
run() { printf '+ %q ' "$@"; printf '\n'; "$@"; }

if [[ "$SKIP_APT" != "1" ]]; then
  log "Installing system dependencies"
  if command -v apt-get >/dev/null 2>&1; then
    run sudo apt-get update
    run sudo apt-get install -y libarchive-tools p7zip-full git curl ca-certificates
  else
    echo "apt-get not found; skipping system package install"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  run bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ "$SKIP_SYNC" != "1" ]]; then
  log "Syncing Python environment"
  run uv sync
fi

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  log "Downloading Lumbras OTB archives"
  run uv run python scripts/download_lumbras_otb.py
fi

if [[ "$SKIP_EXTRACT" != "1" ]]; then
  log "Extracting/splitting 2200+ Lumbras PGNs"
  run uv run python scripts/extract_lumbras_2200_splits.py
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  log "Preprocessing verifier game store"
  run uv run python scripts/preprocess_verifier_dataset.py --workers "$WORKERS" --chunksize "$CHUNKSIZE"
fi

case "$RUN_TRAIN" in
  probe)
    log "Running Q-encoder fact probe"
    run uv run python scripts/train_encoder_q_probe.py \
      --batch-size 32 \
      --context-plies 125 \
      --max-probe-plies 250 \
      --bucket-plies 25 \
      --model-dim 256 \
      --heads 16 \
      --grad-accum-steps 16
    ;;
  qverifier)
    log "Running Q-verifier half-game prefix training"
    run uv run python scripts/train_encoder_q.py \
      --batch-size 32 \
      --sample-mode prefix \
      --prefix-fraction 0.5 \
      --context-plies 125 \
      --min-game-plies 100 \
      --max-game-plies 250 \
      --model-dim 256 \
      --heads 16 \
      --grad-accum-steps 16
    ;;
  none)
    log "Done; not starting training because --run none"
    ;;
  *)
    echo "invalid --run value: $RUN_TRAIN (expected probe, qverifier, none)" >&2
    exit 2
    ;;
esac
