# BSE neighbor explorer

Exploratory UI for asking what strategic information appears in the frozen board-state encoder latent geometry.

## Build index on the target machine

Use the latest interrupted 256d/48Q checkpoint:

```bash
uv run python scripts/build_bse_neighbor_index.py \
  --pgn data/processed/lumbras/<1800_2200_source_1>.pgn data/processed/lumbras/<1800_2200_source_2>.pgn \
  --checkpoint checkpoints/board_state_q_probe_256d_48q/board_state_q_probe_volcanic-planet-22_epoch_001_batch_1350000.pt \
  --out-dir data/analysis/bse_neighbors_1800_2200_100k \
  --nonterminal-samples 90000 \
  --terminal-samples 10000 \
  --min-elo 1800 \
  --max-elo 2200 \
  --bucket-plies 5 \
  --device cuda
```

The builder writes:

```text
data/analysis/bse_neighbors_1800_2200_100k/
  config.json
  metadata.npy
  vectors_raw.fp16.npy
  vectors_norm.fp16.npy
  games/
    moves.npy
    offsets.npy
    pgn_texts.jsonl
    game_headers.jsonl
```

Metadata intentionally stores only:

```text
game_id, ply, is_terminal
```

The UI reconstructs FEN, SAN, legal moves, headers, and analysis at runtime.

## Install Stockfish

```bash
bash install-stockfish.sh
```

Default staged path:

```text
tools/stockfish/stockfish
```

## Serve UI

```bash
uv run python scripts/serve_bse_neighbor_explorer.py \
  --index-dir data/analysis/bse_neighbors_1800_2200_100k \
  --checkpoint checkpoints/board_state_q_probe_256d_48q/board_state_q_probe_volcanic-planet-22_epoch_001_batch_1350000.pt \
  --stockfish-path tools/stockfish/stockfish \
  --device cuda \
  --host 0.0.0.0 \
  --port 8765
```

Open:

```text
http://localhost:8765
```

## Search formulae

Raw L2:

```text
d_i = ||x_i - q||_2^2 = sum_j (x_ij - q_j)^2
```

Cosine:

```text
xhat_i = x_i / max(||x_i||_2, eps)
qhat   = q   / max(||q||_2, eps)
score_i = dot(xhat_i, qhat)
```

The UI returns top 5 neighbors from distinct games and can exclude the query game.
