# How to Run

## Setup

Run commands from the repository root:

```bash
cd /home/amanrai/Code/chess
uv sync
```

If you are not using `uv`, activate an environment with the dependencies in `pyproject.toml` installed, then replace `uv run python` below with `python`.

## Train the Q-Verifier

Requested run: batch size 8, full-game samples, and only games up to 200 move packets/plies:

```bash
uv run python scripts/train_encoder_q.py \
  --batch-size 8 \
  --sample-mode full \
  --max-game-moves 200
```

Notes:

- `--sample-mode full` feeds the full game into the verifier dataset before truncation/padding to `--context-moves`.
- `--sample-mode prefix` samples a random prefix length from prefix buckets and labels it with the final game result.
- `--bucket-mode fraction` chooses equally among first 20%, 20-50%, 50-80%, and final 20% of each game, then samples uniformly inside that bucket.
- `--bucket-mode absolute` chooses equally among plies 1-16, 17-40, 41-80, and 81+.
- `--max-game-moves 200` excludes games longer than 200 move packets/plies; it does not force every prefix to be 200 plies.
- If a sample has more rows than `--context-moves`, the dataset keeps the last `--context-moves` rows.

### Q-Verifier options

Defaults are shown in parentheses.

```text
--data-dir PATH             Verifier game-store directory (data/processed/lumbras/verifier)
--context-moves INT         Number of move packets in model context (128)
--sample-mode {prefix,full} Prefix sampling or full-game samples (prefix)
--bucket-mode {fraction,absolute}
                            Prefix bucket strategy when sample-mode is prefix (fraction)
--max-game-moves INT        Exclude games longer than this many move packets/plies (unset)
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
```

End-of-epoch checkpoints are written as:

```text
checkpoints/q_verifier/q_verifier_epoch_<N>.pt
```

In-epoch snapshots are written every `--snapshot-every-batches` batches by default:

```text
checkpoints/q_verifier/q_verifier_epoch_<EEE>_batch_<BBBBBB>.pt
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
--seq-len INT             Fixed token sequence length per move packet (8)
--max-games INT           Debug limit; 0 means all games (0)
--materialize-prefixes    Also write fixed prefix_x.npy / prefix_y.npy samples
--context-moves INT       Context size for materialized prefixes (128)
--prefixes-per-game INT   Number of materialized prefixes per game (1)
--bucket-mode {fraction,absolute}
                          Prefix bucket strategy for materialized prefixes (fraction)
--seed INT                Prefix sampling seed (0)
--workers INT             Tokenization worker processes (1)
--chunksize INT           ProcessPool chunksize when workers > 1 (64)
--no-progress             Disable progress bars
```

## Train the non-Q Verifier

There is also a baseline verifier trainer:

```bash
uv run python scripts/train_verifier.py
```

### Baseline verifier options

```text
--data-dir PATH             Verifier data directory (data/processed/lumbras/verifier)
--context-moves INT         Number of move packets in context (128)
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

## Tokenize PGN moves only

For a flat move-token array rather than verifier game-store data:

```bash
uv run python scripts/tokenize_pgn_dataset.py
```

Options:

```text
inputs...              Optional PGN input paths. Defaults to the two Lumbras processed PGNs.
--output, -o PATH      Output .npy path (data/processed/lumbras/tokenized_moves_u16_seq8.npy)
--seq-len INT          Fixed token sequence length per move packet (8)
--limit-moves INT      Smoke-test limit; 0 means all moves (0)
--count-only           Count rows without writing the array
```

## Tests

```bash
uv run pytest
```
