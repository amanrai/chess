# Data pipeline order

Run these commands from the repo root.

```bash
cd /home/amanrai/Code/chess
```

## 0. Set up environment

```bash
uv sync
```

## 1. Download Lumbras OTB archives

Downloads compressed `.7z` PGN archives into `data/raw/lumbras/otb/`.

```bash
uv run python scripts/download_lumbras_otb.py
```

Expected output directory:

```text
data/raw/lumbras/otb/
```

## 2. Extract and split 2200+ datasets

Creates the base and fine-tune PGN splits from the downloaded archives.

```bash
uv run python scripts/extract_lumbras_2200_splits.py
```

Expected files:

```text
data/processed/lumbras/lumbras_otb_both_2200_to_2399_base.pgn
data/processed/lumbras/lumbras_otb_both_2400_plus_ft.pgn
data/processed/lumbras/lumbras_otb_2200_splits_manifest.json
```

Counts from local run:

```text
base: 1,999,724 games, both players 2200-2399
ft:     911,091 games, both players 2400+
```

## 3. Preprocess verifier game store

Converts the split PGNs into compact token arrays while preserving game boundaries.

```bash
uv run python scripts/preprocess_verifier_dataset.py
```

For faster preprocessing on a multi-core machine:

```bash
uv run python scripts/preprocess_verifier_dataset.py --workers $(nproc)
```

You can tune process-pool batching:

```bash
uv run python scripts/preprocess_verifier_dataset.py --workers $(nproc) --chunksize 128
```

The script prints progress bars and final dataset stats. Disable progress bars with:

```bash
uv run python scripts/preprocess_verifier_dataset.py --no-progress
```

Expected files:

```text
data/processed/lumbras/verifier/moves.npy
data/processed/lumbras/verifier/offsets.npy
data/processed/lumbras/verifier/results.npy
data/processed/lumbras/verifier/manifest.json
data/processed/lumbras/verifier/sources.json
```

Meaning:

```text
moves.npy    uint16[total_moves, 8]
offsets.npy  int64[num_games + 1]
results.npy  int64[num_games]
```

Labels:

```text
0 = white win
1 = black win
2 = draw
```

Stats written to `manifest.json` include:

```text
result counts
game length min/max/mean/median/p10/p90
source file counts
skipped game counts
workers/chunksize used
```

## 4. Optional: materialize sampled verifier prefixes

Normally we can sample prefixes dynamically during training from `moves.npy + offsets.npy + results.npy`.

If you want a fixed prefix dataset:

```bash
uv run python scripts/preprocess_verifier_dataset.py --materialize-prefixes --context-moves 128 --prefixes-per-game 4
```

Expected extra files:

```text
data/processed/lumbras/verifier/prefix_x.npy
data/processed/lumbras/verifier/prefix_y.npy
data/processed/lumbras/verifier/prefix_meta.jsonl
data/processed/lumbras/verifier/prefix_manifest.json
```

## Quick debug run

Use `--max-games` to test quickly without processing everything:

```bash
uv run python scripts/preprocess_verifier_dataset.py --max-games 100
```

Optional fixed prefixes on a small debug set:

```bash
uv run python scripts/preprocess_verifier_dataset.py --max-games 100 --materialize-prefixes --context-moves 64 --prefixes-per-game 2
```

## Important notes

- `data/` is gitignored.
- Raw PGNs and processed arrays should not be committed.
- Full dataset tokenization/preprocessing can be rerun on a Tailnet/Jupyter box before moving compact arrays to Vast.
- Vast should generally receive compact tokenized arrays, not raw PGNs or Lumbras archives.
