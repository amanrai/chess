# Handoff — 28 Jul

## Current repo state

Current branch / pushed HEAD during this work:

```text
main
093b89c Report board-state class stats per print interval
```

Recent relevant commits:

```text
093b89c Report board-state class stats per print interval
fd2dae8 Reduce board-state training print frequency
0609ee0 Resume board-state training deterministically
57a7089 Move small BSE config to board-state trainer
3a46d87 Shrink q-probe training config
```

Important note: `3a46d87` mistakenly applied the small/bf16 changes to the older `q_probe` fact-probe path. `57a7089` corrected this by restoring the q-probe path and moving the intended changes to the board-state trainer. The current working-tree diff for committed files should be clean. Local untracked files observed:

```text
handoff_23_jul.md
uv.lock
handoff_28_jul.md  # this file
```

## Board-state encoder / BSE shrink experiment

The board-state encoder is now being aggressively shrunk because it is the foundation for all downstream manifold, diffusion, trajectory, and move-selection work. The original successful BSE was roughly:

```text
model_dim:      512
heads:          32
history_layers: 6
q_layers:       4
num_queries:    64
encoder params: ~31.6M
```

The active preferred small run is:

```text
model_dim:      256
heads:          16
history_layers: 6
q_layers:       4
num_queries:    48
batch_size:     8
grad_accum:     64
effective batch: 512
dtype:          bf16 model params/activations, float32 CE logits
checkpoint dir: checkpoints/board_state_q_probe_256d_48q
```

Rationale for `num_queries=48`: deliberately avoid the appearance that `64` latent slots correspond one-to-one with the 64 board squares. The square identity enters only through the separate learned square-query embedding table in the probe head.

Approximate parameter scaling:

```text
512d encoder -> ~32M
256d encoder -> ~8M
128d encoder -> ~2M
64d encoder  -> ~0.5M
```

The working hypothesis after current experiments is that `256d/48Q` is clearly sufficient and may be the practical new foundation. `128d` may still get there with much more data. `64d` is not currently being pursued.

## Active board-state trainer configuration

Use:

```bash
bash train-board-state-verifier.sh
```

Current `train-board-state-verifier.sh` important settings:

```text
DATA_DIR="data/processed/lumbras/verifier"
BOARD_STATE_DIR="data/processed/lumbras/verifier/board_state"
CONTEXT_PLIES="128"
SQUARES_PER_POSITION="16"
BUCKET_PLIES="25"
BATCH_SIZE="8"
GRAD_ACCUM_STEPS="64"
EPOCHS="1"
LEARNING_RATE="3e-4"
WEIGHT_DECAY="0.01"
ENCODER_INTERNAL_DIM="256"
ENCODER_ATTENTION_HEADS="16"
ENCODER_HISTORY_LAYERS="6"
Q_PROBE_LAYERS="4"
Q_PROBE_QUERY_SLOTS="48"
DROPOUT="0.0"
DATALOADER_WORKERS="4"
DEVICE="cuda"
CHECKPOINT_DIR="checkpoints/board_state_q_probe_256d_48q"
SNAPSHOT_EVERY_BATCHES="10000"
LOG_WINDOW="1000"
PRINT_EVERY_BATCHES="100"
SEED="0"
WANDB_FLAG="--wandb"
```

The shell script does not currently append arbitrary `"$@"` extra args. If a one-off override is needed, either edit the script or run `scripts/train_board_state_q_probe.py` directly.

## Board-state trainer changes made today

File: `scripts/train_board_state_q_probe.py`

### bf16

The board-state trainer now moves the model to bf16:

```python
model = model.to(device=args.device, dtype=torch.bfloat16)
```

Cross-entropy uses float32 logits for stability:

```python
board_loss = F.cross_entropy(logits.float().reshape(-1, NUM_OCCUPANTS), labels.reshape(-1))
check_loss = F.cross_entropy(check_logits.float(), check_y)
mate_loss = F.cross_entropy(mate_logits.float(), mate_y)
turn_loss = F.cross_entropy(turn_logits.float(), turn_y)
```

RoPE was updated in `src/chessgm/network.py` so cos/sin are cast back to activation dtype:

```python
cos = emb.cos().to(dtype=x.dtype)
sin = emb.sin().to(dtype=x.dtype)
```

### deterministic resume

Resume is enabled by default:

```python
parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, ...)
parser.add_argument("--resume-checkpoint", type=Path, default=None, ...)
parser.add_argument("--seed", type=int, default=0, ...)
```

The trainer resumes from the latest checkpoint in `--checkpoint-dir` unless `--no-resume` is passed or an explicit `--resume-checkpoint` is used.

Checkpoints now include:

```text
model
optimizer
args
epoch
batch
global_batch
epoch_sample_offset
run_id
```

A deterministic `AffineShuffleSampler` was added. It uses a seed-derived affine permutation and starts at the saved `epoch_sample_offset`, so a resumed run should not replay already-seen data. This was validated in practice after a power interruption: the run resumed around the expected batch/sample offset and metrics continued smoothly instead of restarting.

Snapshot filenames continue to use absolute epoch batch ids, e.g.:

```text
board_state_q_probe_<run_id>_epoch_001_batch_330000.pt
```

`tqdm` may show only remaining loader length after resume, but the printed `batch=X/num_batches` and checkpoint filename batch ids are the absolute epoch batch counters and are the ones to trust.

### print interval class stats

Printed occupant-class `n` and `pred_n` stats were initially rolling over `--log-window`, which made them monotonically increase until the log window filled. This was changed to per-print-interval counters that reset after each print. With `PRINT_EVERY_BATCHES=100`, the class table now reflects the last 100 batches.

## Board-state probe architecture wording

Current accurate description:

### Encoding

Each ply is represented as an eight-slot token packet. `CHECK` and `MATE` tokens are stripped before encoding to avoid leakage, and remaining slots are padded as needed. Within a ply, token-slot order is encoded using learned positional embeddings. Across plies, temporal order is encoded using RoPE indexed by ply number.

A variable-length move-history prefix is sampled from the game record, padded/truncated to the context window, embedded, and processed by a six-layer pre-norm Transformer encoder with custom RoPE self-attention.

A learned Q-former query bank of shape `(48, model_dim)` cross-attends over the encoder output, using learned query parameters as queries and encoded history representations as keys/values. This produces fixed-size latent representation:

```text
z1 ∈ R^(48 × model_dim)
```

The number of query slots is deliberately 48 rather than 64, so the latent bank cannot be trivially interpreted as one slot per board square.

### Querying

A separate learned square embedding table of shape `(64, model_dim)` maps each board square to a fixed learned query vector. The ith embedding always corresponds to the same board square.

For each training example, 16 board squares are sampled uniformly at random. Each selected square embedding cross-attends to `z1`, using the square embedding as query and `z1` as keys/values. The attended square representation goes through a residual MLP and linear classifier, producing logits over 13 classes: empty plus the 12 colored piece types. Training uses cross-entropy directly on logits.

The board-state head is `DiffThinkerBoardStateQueryMLP` in `src/chessgm/network_q.py`:

```text
square_embedding: Embedding(64, D)
query/context LayerNorm
single-head cross-attention square queries -> q_bank
residual MLP: Linear(D, 4D), GELU, Dropout, Linear(4D, D), Dropout
final LayerNorm
Linear(D, 13)
```

## Observed 256d/48Q training results

These are sampled-square training probe metrics, not held-out all-64 eval. Held-out eval still needs to be run after the run matures.

At ~13.2% seen:

```text
batch:              330400/2500000
board_loss:         0.0881
square_acc:         0.9693
occupied_precision: 0.9695
occupied_recall:    0.9457
```

At ~15.5% seen:

```text
batch:              386700/2500000
board_loss:         0.0713
square_acc:         0.9758
occupied_precision: 0.9756
occupied_recall:    0.9603
```

At ~28.7% seen:

```text
batch:              717800/2500000
board_loss:         0.0390
square_acc:         0.9872
occupied_precision: 0.9901
occupied_recall:    0.9808
```

At ~31.1% seen:

```text
batch:              776300/2500000
board_loss:         0.0364
square_acc:         0.9884
occupied_precision: 0.9864
occupied_recall:    0.9761
```

Piece-level stats at ~31% were excellent. Most piece precision/recall were in the high 0.96–0.99 range; rooks remained the relative weak spot but were still good:

```text
WHITE_ROOK precision/recall: 0.980 / 0.908
BLACK_ROOK precision/recall: 0.945 / 0.917
```

Well-sampled ply buckets at ~31%:

```text
1-25:    sq_acc 1.000, occ_p 1.000, occ_r 1.000
26-50:   sq_acc 0.992, occ_p 0.986, occ_r 0.992
51-75:   sq_acc 0.978, occ_p 0.967, occ_r 0.952
76-100:  sq_acc 0.971, occ_p 0.975, occ_r 0.896
101-125: sq_acc 0.991, occ_p 0.981, occ_r 0.944
126-150: sq_acc 0.967, occ_p 0.935, occ_r 0.784  # low N=17
```

Interpretation: `256d/48Q` is clearly sufficient for the sampled training distribution and looks like the likely new default BSE foundation if held-out all-64 eval confirms it.

## 128d experiment

A separate 128d run is also training, reportedly with:

```text
model_dim: 128
heads:     8
num_queries: 48
```

A 128d/16-head config was considered, but the user hit a size mismatch, so 128d/8h is being used for now. This preserves head dimension 16:

```text
128 / 8 = 16
```

At ~24% data seen, 128d metrics were:

```text
batch:              304500/1250000
board_loss:         0.5338
square_acc:         0.8352
occupied_precision: 0.8192
occupied_recall:    0.6462
```

At ~28% data seen:

```text
batch:              350000/1250000
board_loss:         0.4702
square_acc:         0.8552
occupied_precision: 0.8598
occupied_recall:    0.6750
```

128d is much slower than 256d and still struggles with deep/midgame occupancy recall, but its loss curve is clearly decreasing rather than plateauing. It may eventually work with more data. The plan is to let 128d run for several more days if it continues improving.

## 64d experiment

64d was discussed but is not currently being pursued. Approximate encoder size would be ~0.5M params. Given 128d is already much slower than 256d, 64d is considered mostly a curiosity for now.

## Game-length histogram note

The board-state training bucket table is not a final game-length histogram. It reports sampled prefix positions. The preprocessing samples prefixes using eligible-game bucket weighting:

```text
eligible_games_for_bucket = games with length >= bucket_start
allocation ∝ eligible_games ^ allocation_alpha
```

with default `allocation_alpha=1.15`, which favors earlier buckets. Therefore the bucket table cannot be used to conclude most games end around a given ply count.

Actual game length histogram script:

```bash
python scripts/histogram_game_plies.py \
  --data-dir data/processed/lumbras/verifier
```

It reads `offsets.npy` and computes:

```python
lengths = np.diff(offsets)
```

Optional JSON output:

```bash
python scripts/histogram_game_plies.py \
  --data-dir data/processed/lumbras/verifier \
  --output data/processed/lumbras/verifier/game_length_histogram.json
```

## Scryer ticket update: Manifolder-tron 3000

The manifold separator ticket was renamed and rewritten.

Current ticket:

```text
Title: Manifolder-tron 3000
ID: bb06ffbb-ae49-4616-9e19-8f9d2d01a90e
Status: in_execution
```

The ticket now reflects the more ambitious combined objective:

```text
joint outcome manifold + local trajectory field + move reconstruction
```

rather than only:

```text
terminal separator first, vector field later
```

Core idea: freeze BSE, train bank-shaped `Z`, optional result readout `M`, move decoder/inverse-transition head, and trajectory/diffusion-prep heads.

The new ticket description includes four loss families:

```text
L_total = α_terminal(t)  * L_terminal
        + β_action(t)    * L_action_reconstruction
        + γ_direction(t) * L_local_direction
        + δ_triplet(t)   * L_manifold_separation
```

Loss weights should be configurable, nonzero, and possibly cyclic/scaled-sine scheduled to avoid catastrophic forgetting while shifting emphasis over time.

### Manifolder notation from ticket

For a randomly picked starting point:

```text
h_tN = randomly picked starting point from game log; encoder sees h_0:h_t
h_tt = terminal state from same game log
h_next_ply = next ply after h_tN; what opposing player did
h_next_ply_same_player = h_next_ply + 1; what same player did after opponent moved
```

For similar-length opposite-winner games:

```text
a = h_t0_white
b = h_t0_black
w_white = h_tt_white
w_black = h_tt_black

a_next_ply, b_next_ply
a_next_ply_same_player, b_next_ply_same_player
```

Candidate losses:

- terminal state separation / classification
- action reconstruction from latent transitions
- local direction consistency, e.g. `1 - cosine_similarity(v1, v2)`
- triplet/margin manifold separation using terminal anchors and intermediate states

Important conceptual point: when later training forward trajectory / diffusion, retain some losses from Manifolder-tron to prevent catastrophic forgetting. Either freeze `Z` for the first diffusion proof or continue to train `Z` with retained nonzero manifold/action/direction losses.

## Next recommended steps

1. Let `256d/48Q` continue training locally for at least another day or two. There is no marginal cloud cost, and it is still improving.
2. Let `128d/48Q` continue for several days if it keeps moving downward; decide whether it has a delayed phase transition or is capacity-limited.
3. After a strong 256d checkpoint is available, run held-out all-64 eval against `data/processed/lumbras/eval_board_state_2000_2199`:

```bash
uv run python scripts/evaluate_board_state_q_probe.py \
  --eval-dir data/processed/lumbras/eval_board_state_2000_2199 \
  --max-games 100 \
  --batch-size 32 \
  --checkpoint checkpoints/board_state_q_probe_256d_48q/<latest>.pt
```

Compare to original 512d metrics:

```text
exact_square: 0.9907
exact_board:  0.6381
avg_wrong_sq: 0.598
p50_wrong:    0
p75_wrong:    1
p90_wrong:    2
occ_p/occ_r:  0.9895 / 0.9845
```

4. If 256d held-out eval is close, declare `256d/48Q` the default frozen BSE for Manifolder-tron.
5. Consider adding W&B resume support later. Training resume works, but W&B currently may create a new run after interruption. Future fix: store `wandb_run_id` in checkpoints, peek checkpoint before `wandb.init`, and call `wandb.init(id=wandb_run_id, resume="allow")`.
