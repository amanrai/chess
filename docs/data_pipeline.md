# Data pipeline order

Run these commands from the repo root.

```bash
cd /home/amanrai/Code/chess
```

## 0. Set up environment

```bash
uv sync
```

## Vast one-shot download/preprocess/run

On a fresh Vast instance, clone the repo and run:

```bash
bash scripts/vast_download_preprocess_run.sh
```

That script installs system dependencies when `apt-get` is available, installs/syncs `uv`, downloads Lumbras archives, extracts the 2200+ PGN splits, builds the verifier game store, then starts the Q-encoder fact probe. If `WANDB_API_KEY` is set or `.env` contains `wandb_key=...`, W&B logging is enabled for the probe run.

Useful variants:

```bash
# Download and preprocess only; do not start training.
bash scripts/vast_download_preprocess_run.sh --run none

# Run the half-game Q-verifier after preprocessing instead of the fact probe.
bash scripts/vast_download_preprocess_run.sh --run qverifier

# Resume after data is already downloaded/extracted.
bash scripts/vast_download_preprocess_run.sh --skip-download --skip-extract

# Tune preprocessing workers/chunksize.
bash scripts/vast_download_preprocess_run.sh --workers 32 --chunksize 256

# Force W&B on/off for the probe run.
bash scripts/vast_download_preprocess_run.sh --wandb 1
bash scripts/vast_download_preprocess_run.sh --wandb 0
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
moves.npy    uint16[total_plies, 8]  # filename is historical; rows are plies
offsets.npy  int64[num_games + 1]
results.npy  int64[num_games]
```

Labels:

```text
0 = white win
1 = black win
2 = draw
```

## 3a. Build board-state verifier targets and probe sampling plan

After building the game store, replay its exact source PGNs into compact
post-ply board labels plus a 20M-position sampling plan for the board-state
probe:

```bash
uv sync  # installs python-chess
uv run python scripts/preprocess_board_state_verifier.py
```

This writes `data/processed/lumbras/verifier/board_state/`:

```text
board_after_packed.npy  uint8[total_plies, 32]  # 64 four-bit occupant labels
probe_samples.npy       uint32[20_000_000, 2]   # (game_index, prefix_plies)
manifest.json
```

The sampler chooses a prefix-position bucket first, samples uniformly from
games that reach that position, and uses an eligible-game-count exponent
(default `--allocation-alpha 1.15`) to give early positions extra budget. It
therefore preserves the natural mix of short and long games in early positions
without repeatedly over-sampling the rare long-game cohort. `--max-prefix-plies
0` includes the full game-length tail.

The script checks every accepted PGN game's result and token packets against
`moves.npy` before writing its board labels. Supply the exact inputs and order
used for the game store if they differ from the defaults.

Stats written to `manifest.json` include:

```text
result counts
game length min/max/mean/median/p10/p90
source file counts
skipped game counts
workers/chunksize used
```

## 4. Optional: materialize sampled verifier prefixes

Normally we sample prefixes dynamically during training from `moves.npy + offsets.npy + results.npy`.
The dynamic Q-verifier trainer supports controlled percentage-based prefixes, e.g. `--prefix-fraction 0.5`, plus game-length filters like `--min-game-plies` and `--max-game-plies`.

Important distinction:

- prefix fraction / prefix length: how much of the original game is sampled
- context plies: how many sampled ply packets the model can see after crop/pad

Use percentage-based prefixes for interpretable partial-game probes. A fixed context window by itself can include whole short games.

Encoder fact probes can also read directly from this dynamic game store. The check/mate probe derives labels from `CHECK`/`MATE` tokens, then removes those tokens from the input side online to avoid label leakage; no separate preprocessing step is required.

If you want a fixed prefix dataset:

```bash
uv run python scripts/preprocess_verifier_dataset.py --materialize-prefixes --context-plies 128 --prefixes-per-game 4
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
uv run python scripts/preprocess_verifier_dataset.py --max-games 100 --materialize-prefixes --context-plies 64 --prefixes-per-game 2
```

## Important notes

- `data/` is gitignored.
- Raw PGNs and processed arrays should not be committed.
- Full dataset tokenization/preprocessing can be rerun on a Tailnet/Jupyter box before moving compact arrays to Vast.
- Vast should generally receive compact tokenized arrays, not raw PGNs or Lumbras archives.
