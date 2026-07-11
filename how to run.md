# How to Run

## Setup

Run commands from the repository root:

```bash
cd /home/amanrai/Code/chess
uv sync
```

If you are not using `uv`, activate an environment with the dependencies in `pyproject.toml` installed, then replace `uv run python` below with `python`.

## Train the Q-Verifier

Requested run: batch size 8, full-game samples, and only games up to 200 plies:

```bash
uv run python scripts/train_encoder_q.py \
  --batch-size 8 \
  --sample-mode full \
  --max-game-plies 200
```

Notes:

- `--sample-mode full` feeds the full game into the verifier dataset before truncation/padding to `--context-plies`.
- `--sample-mode prefix` samples an incomplete prefix and labels it with the final game result.
- For controlled partial-game experiments, prefer `--prefix-fraction`, e.g. `--prefix-fraction 0.5` for exactly the first half of each sampled game.
- `--prefix-fraction-min` / `--prefix-fraction-max` sample a random percentage range, e.g. `0.4` to `0.6`.
- If no explicit prefix fraction is set, `--bucket-mode fraction` chooses equally among first 20%, 20-50%, 50-80%, and final 20% of each game, then samples uniformly inside that bucket.
- `--bucket-mode absolute` chooses equally among plies 1-16, 17-40, 41-80, and 81+.
- `--max-game-plies 200` excludes games longer than 200 plies; it does not force every prefix to be 200 plies.
- `--min-game-plies` excludes very short games if you do not want tiny games dominating or becoming trivial.
- If a sample has more rows than `--context-plies`, the dataset keeps the last `--context-plies` rows.

### Half-game prefix example

This uses the first 50% of each sampled game, filters to games with 100-250 plies, and keeps enough context to avoid truncating half of a 250-ply game:

```bash
uv run python scripts/train_encoder_q.py \
  --batch-size 32 \
  --sample-mode prefix \
  --prefix-fraction 0.5 \
  --context-plies 125 \
  --min-game-plies 100 \
  --max-game-plies 250 \
  --model-dim 256 \
  --heads 16 \
  --grad-accum-steps 16
```

### Q-Verifier options

Defaults are shown in parentheses.

```text
--data-dir PATH             Verifier game-store directory (data/processed/lumbras/verifier)
--context-plies INT         Number of ply packets in model context (128)
--sample-mode {prefix,full} Prefix sampling or full-game samples (prefix)
--bucket-mode {fraction,absolute}
                            Prefix bucket strategy when sample-mode is prefix and no explicit prefix fraction is set (fraction)
--min-game-plies INT        Exclude games shorter than this many plies (unset)
--max-game-plies INT        Exclude games longer than this many plies (unset)
--prefix-fraction FLOAT     Use exactly this fraction of each game as the prefix, e.g. 0.5 for half-game (unset)
--prefix-fraction-min FLOAT Minimum random prefix fraction; used with --prefix-fraction-max (unset)
--prefix-fraction-max FLOAT Maximum random prefix fraction; used with --prefix-fraction-min (unset)
--batch-size INT            Per-step batch size (32)
--grad-accum-steps INT      Gradient accumulation steps; effective batch = batch-size * grad-accum-steps (8)
--epochs INT                Training epochs (1)
--lr FLOAT                  Learning rate (3e-4)
--weight-decay FLOAT        AdamW weight decay (0.01)
--model-dim INT             Transformer width (256)
--heads INT                 Attention heads (8)
--history-layers INT        History encoder layers (4)
--q-layers INT              Query/cross-attention layers (2)
--num-queries INT           Learned query tokens (16)
--dropout FLOAT             Dropout probability (0.0)
--examples-per-epoch INT    Override examples per epoch; default is one per valid game (unset)
--num-workers INT           DataLoader workers (0)
--device STR                Device, e.g. cuda, cuda:0, cpu (cuda if available else cpu)
--checkpoint-dir PATH       Output checkpoint directory (checkpoints/q_verifier)
--snapshot-every-batches INT
                            Save an in-epoch snapshot every N batches; 0 disables (5000)
--log-window INT            Rolling metric window in batches (1000)
--wandb                    Enable Weights & Biases logging; reads WANDB_API_KEY or .env wandb_key
--wandb-project STR         W&B project (chess-gm)
--wandb-run-name STR        W&B run name (unset)
--wandb-log-every INT       W&B logging interval in batches (100)
```

End-of-epoch checkpoints are written as:

```text
checkpoints/q_verifier/q_verifier_epoch_<N>.pt
```

In-epoch snapshots are written every `--snapshot-every-batches` batches by default:

```text
checkpoints/q_verifier/q_verifier_epoch_<EEE>_batch_<BBBBBB>.pt
```

## Vast One-Shot Data Prep + Run

On a fresh Vast instance, clone the repo and run:

```bash
bash scripts/vast_download_preprocess_run.sh
```

Default behavior: install deps, `uv sync`, download Lumbras, extract/split PGNs, preprocess the verifier game store, then start the Q-encoder fact probe. If `.env` contains `wandb_key=...` or `WANDB_API_KEY` is set, the Vast script enables W&B for the probe run automatically.

Useful variants:

```bash
bash scripts/vast_download_preprocess_run.sh --run none
bash scripts/vast_download_preprocess_run.sh --run qverifier
bash scripts/vast_download_preprocess_run.sh --skip-download --skip-extract
bash scripts/vast_download_preprocess_run.sh --workers 32 --chunksize 256
```

## Build the Verifier Game Store

The Q-verifier expects preprocessed game-store arrays:

```text
moves.npy
offsets.npy
results.npy
manifest.json
```

Create them from the default Lumbras PGN inputs:

```bash
uv run python scripts/preprocess_verifier_dataset.py --workers $(nproc)
```

### Preprocessing options

```text
inputs...                 Optional PGN input paths. Defaults to the two Lumbras processed PGNs.
--out-dir PATH            Output directory (data/processed/lumbras/verifier)
--seq-len INT             Fixed token sequence length per ply packet (8)
--max-games INT           Debug limit; 0 means all games (0)
--materialize-prefixes    Also write fixed prefix_x.npy / prefix_y.npy samples
--context-plies INT       Context size for materialized prefixes (128)
--prefixes-per-game INT   Number of materialized prefixes per game (1)
--bucket-mode {fraction,absolute}
                          Prefix bucket strategy for materialized prefixes (fraction)
--seed INT                Prefix sampling seed (0)
--workers INT             Tokenization worker processes (1)
--chunksize INT           ProcessPool chunksize when workers > 1 (64)
--no-progress             Disable progress bars
```

## Train Q-Encoder Fact Probes

Use this to test whether the encoder can recover simple facts from prefixes without relying on final WDL labels.

The probe samples a random prefix length up to `--max-probe-plies` and predicts:

- whether the final ply in the prefix resulted in check/mate
- whose turn it is next

Important: `CHECK` and `MATE` tokens are removed from the input `x` and used only to build the check/mate labels, so the model cannot read the answer directly.

```bash
uv run python scripts/train_encoder_q_probe.py \
  --batch-size 32 \
  --context-plies 125 \
  --max-probe-plies 250 \
  --model-dim 256 \
  --heads 16 \
  --wandb
```

The probe logs all printed metrics to W&B when `--wandb` is set, including loss splits, accuracies, positive-class probabilities, recalls, raw counts, and per-ply-bucket metrics. It reads the key from `WANDB_API_KEY` or `.env`:

```text
wandb_key=...
```

Optional: initialize the shared encoder from a Q-verifier checkpoint:

```bash
uv run python scripts/train_encoder_q_probe.py \
  --init-checkpoint checkpoints/q_verifier/q_verifier_epoch_1.pt
```

## Train the non-Q Verifier

There is also a baseline verifier trainer:

```bash
uv run python scripts/train_verifier.py
```

### Baseline verifier options

```text
--data-dir PATH             Verifier data directory (data/processed/lumbras/verifier)
--context-plies INT         Number of ply packets in context (128)
--batch-size INT            Batch size (32)
--epochs INT                Training epochs (1)
--lr FLOAT                  Learning rate (3e-4)
--weight-decay FLOAT        Weight decay (0.01)
--model-dim INT             Transformer width (256)
--heads INT                 Attention heads (8)
--layers INT                Transformer layers (6)
--dropout FLOAT             Dropout probability (0.0)
--examples-per-epoch INT    Override examples per epoch (unset)
--num-workers INT           DataLoader workers (0)
--device STR                Device override (unset)
--checkpoint-dir PATH       Output checkpoint directory (checkpoints/verifier)
--log-window INT            Rolling metric window in batches (1000)
--wandb                    Enable Weights & Biases logging
--wandb-project STR         W&B project (chess-gm)
--wandb-run-name STR        W&B run name (unset)
--wandb-log-every INT       W&B logging interval in batches (10)
```

## Tokenize PGN plies only

For a flat ply-token array rather than verifier game-store data:

```bash
uv run python scripts/tokenize_pgn_dataset.py
```

Options:

```text
inputs...              Optional PGN input paths. Defaults to the two Lumbras processed PGNs.
--output, -o PATH      Output .npy path (data/processed/lumbras/tokenized_moves_u16_seq8.npy)
--seq-len INT          Fixed token sequence length per ply packet (8)
--limit-plies INT      Smoke-test limit; 0 means all plies (0)
--count-only           Count rows without writing the array
```

## Tests

```bash
uv run pytest
```
