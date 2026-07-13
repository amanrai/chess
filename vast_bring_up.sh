#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python scripts/download_lumbras_otb.py
python scripts/extract_lumbras_2200_splits.py
python scripts/preprocess_verifier_dataset.py --workers "$(nproc)"
