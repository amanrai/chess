# Handoff — 29 Jul

## Current repo state

Current branch / pushed HEAD during this work:

```text
main
5bd14f9 Show exact prefix match in BSE inspection
```

Recent relevant commits:

```text
5bd14f9 Show exact prefix match in BSE inspection
4a04b39 Rename BSE inspection UI title
c455fdf Display L2 search as RMS distance
598a8f5 Show last move match in neighbor analysis
d179920 Show next player in BSE neighbor explorer
672dc60 Compare Stockfish only for selected neighbor
db9893f Add synchronized ply navigation
f66ce4c Analyze BSE neighbors after search
9f7a2e3 Reduce BSE neighbor encoding batch size
144b3c0 Log accepted games during BSE neighbor build
212b850 Limit BSE neighbor build to 100k games
ab941c7 Cap BSE neighbor build game scan
288a097 Autodiscover BSE neighbor index in launcher
cbfb9a1 Add BSE neighbor explorer launcher
9fce5b3 Create 1800-2200 PGN for BSE neighbor index
8b3b13c Make BSE neighbor launcher accept PGNs
ec00c6c Default BSE neighbor launcher to eval PGN
4bfa1e4 Add BSE neighbor index launcher
9a5ca59 Add BSE neighbor explorer
```

Untracked local files observed:

```text
handoff_23_jul.md
handoff_28_jul.md
handoff_29_jul.md  # this file
uv.lock
```

## BSE milestone conclusion

The board-state encoder (`z1`) should now be treated as substantially evidenced / complete for its intended module role.

Active checkpoint used for inspection after power loss:

```text
checkpoints/board_state_q_probe_256d_48q/board_state_q_probe_volcanic-planet-22_epoch_001_batch_1350000.pt
```

The 256d/48Q bf16 BSE run had reached about 1.35M / 2.5M batches before interruption, with sampled train metrics around:

```text
board_loss:         0.0249
square_acc:         0.9924
occupied_precision: 0.9912
occupied_recall:    0.9857
```

Interpretation agreed during the session:

- `256d/48Q` at bf16 is basically enough for the BSE.
- The old `512d/64Q` / fp32-ish foundation is likely unnecessary.
- BSE has done what it was designed to do: provide a compact frozen board-state sensor.
- The remaining research hypothesis is not state sensing, but whether a planning manifold can be induced on top of the BSE.

Notation established:

```text
z1 = frozen BSE output
history prefix h_0:h_t -> BSE -> z1_bank ∈ R^(48 × 256)
```

## Board State Encoder Inspection tool

A new exploratory UI/backend was added to inspect nearest neighbors in z1 space.

Main files:

```text
build_bse_neighbor_index.sh
serve_bse_neighbor_explorer.sh
install-stockfish.sh
scripts/build_bse_neighbor_index.py
scripts/serve_bse_neighbor_explorer.py
tools/bse_neighbor_explorer/index.html
tools/bse_neighbor_explorer/README.md
```

Dependencies added:

```text
fastapi
uvicorn
```

### Build index

The launcher now creates the 1800-2200 PGN split if missing, then builds the index:

```bash
./build_bse_neighbor_index.sh
```

It creates / uses:

```text
data/processed/lumbras/lumbras_otb_both_1800_to_2200_base.pgn
```

via:

```bash
uv run python scripts/extract_lumbras_2200_splits.py \
  --raw-dir data/raw/lumbras/otb \
  --out-dir data/processed/lumbras \
  --min-elo 1800 \
  --ft-elo 2201
```

Then builds:

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

Index sampling:

```text
100,000 total indexed positions
90,000 non-terminal positions, roughly equal across 5-ply buckets
10,000 terminal positions, sampled across terminal-length buckets
```

Game collection cap:

```text
BSE_NEIGHBOR_MAX_GAMES default: 100000 accepted games
```

Important: the progress bar's `game` count is scanned PGN records, not accepted games. It now logs accepted count in the postfix and prints when the cap is reached, e.g.:

```text
accepted game cap reached: 100,000/100,000
```

Encoding batch size was reduced after OOM:

```text
BSE_NEIGHBOR_BATCH_SIZE default: 16
```

Reason for prior OOM: batch 256 with full self-attention tried to allocate roughly:

```text
B × H × N × N = 256 × 16 × 1024 × 1024 bf16 ≈ 8 GiB
```

If needed:

```bash
BSE_NEIGHBOR_BATCH_SIZE=8 ./build_bse_neighbor_index.sh
```

### Serve UI

Install/stage Stockfish if desired:

```bash
bash install-stockfish.sh
```

Serve:

```bash
./serve_bse_neighbor_explorer.sh
```

Defaults:

```text
index:     data/analysis/bse_neighbors_1800_2200_100k
checkpoint: checkpoints/board_state_q_probe_256d_48q/board_state_q_probe_volcanic-planet-22_epoch_001_batch_1350000.pt
stockfish: tools/stockfish/stockfish
host:      0.0.0.0
port:      8765
device:    cuda
```

Override examples:

```bash
BSE_NEIGHBOR_PORT=9000 ./serve_bse_neighbor_explorer.sh
STOCKFISH_PATH=/path/to/stockfish ./serve_bse_neighbor_explorer.sh
BSE_NEIGHBOR_DEVICE=cpu ./serve_bse_neighbor_explorer.sh
```

Open:

```text
http://localhost:8765
```

or from another machine:

```text
http://<pop-os-bangalore-ip>:8765
```

## UI behavior as of latest commit

Title:

```text
Board State Encoder Inspection
```

Core UI:

- left board: sampled/query game position
- right board: selected nearest-neighbor result
- sample mode:
  - pure random
  - bucket dropdown
- distance modes:
  - cosine
  - RMS
- `exclude same game` toggle
- Search returns top 5 neighbors from distinct games.
- Neighbor pills allow selecting among top 5.
- A single shared `prev ply` and `next ply` advances/rewinds both visible games together.
- Manual per-board ply jumps are still available.

Displayed game metadata includes:

```text
game
ply / terminal_ply
next to play  # computed from ply count, not network output
last
next
check
legal
fen
```

`next to play` convention:

```text
odd ply  -> black
even ply -> white
```

## Stockfish / analysis features

The backend has python-chess + Stockfish analysis endpoints.

After search, Stockfish comparison now runs only for:

```text
query + selected neighbor
```

If a different neighbor pill is selected, comparison reruns automatically for that selected neighbor.

The comparison table includes:

```text
row
game / ply
last
last match
prefix match
eval
best
played next
rank
swing
pv
```

Definitions:

- `last`: `last_move_san`, the move that produced the displayed board.
- `last match`: whether query and selected neighbor have identical `last_move_san`.
- `prefix match`: whether the exact UCI move prefix lists match.
- `eval`: Stockfish score from White's perspective.
  - positive = White better
  - negative = Black better
- `best`: Stockfish best move.
- `played next`: actual next move in the game log.
- `rank`: where the played next move appears in Stockfish MultiPV top moves.
- `pv`: principal variation, displayed as UCI moves.

Known caution:

- The `swing` metric is still suspicious/sign-sensitive because it compares White-perspective eval before/after the played move while side-to-move changes. Do not over-trust it yet.

## Distance metric interpretation

For flattened z1:

```text
z1_bank [48,256] -> flat [12288]
```

Cosine:

```text
xhat_i = x_i / max(||x_i||_2, eps)
qhat   = q   / max(||q||_2, eps)
score_i = dot(xhat_i, qhat)
```

RMS displayed for L2 search:

```text
l2_sq = sum_j (x_ij - q_j)^2
rms   = sqrt(l2_sq / 12288)
```

Ranking is unchanged relative to squared L2 / Euclidean L2 because sqrt and division by a fixed dimension preserve order.

RMS was chosen because it is more intuitive for future planning-manifold distance objectives: average per-latent-dimension difference.

## Observations from z1 inspection

Important qualitative results observed in the UI:

1. Cosine and RMS often return the same or very similar top-1 neighbors.
2. This suggests z1 geometry is coherent; norms are not completely dominating retrieval.
3. z1 nearest neighbors are often visually / structurally similar board states.
4. z1 is not merely indexing off exact prefix:

```text
prefix match: often no
```

5. z1 is not merely indexing off last move:

```text
last match: often no
```

6. z1 can retrieve similar board states with different players to move.
7. z1 can retrieve visually/materially similar positions with very different Stockfish evals.
8. Conversely, some z1-near pairs have similar evals despite different exact prefixes / last moves.

Interpretation:

```text
z1-nearness ≈ board-state / structural similarity
```

but not necessarily:

```text
objective equivalence
tactical value equivalence
same future
same best move
same outcome manifold
```

This is viewed as a good result, not a failure. It means BSE is doing its job as a state sensor. The gap between structural similarity and objective/planning similarity is exactly what `z2` / Manifolder should learn.

## Planning manifold / Manifolder-tron discussion

The next representation is now called:

```text
z2 = planning manifold representation
```

Stack:

```text
moves/history -> frozen BSE -> z1 -> trainable planning manifold Z -> z2
```

Proposed Manifolder 1000 architecture:

```text
z1_bank [48, 256]
  -> per-slot dimensionality-reducing MLP
       256 -> 128 -> 64 -> 64
  -> compressed context [48, 64]
  -> Q-former with learned queries [16, 64]
       queries attend to compressed z1 context as keys/values
  -> z2_bank [16, 64]
```

No reconstruction/decompression path for Manifolder 1000. That may be considered in later versions if z2 loses too much state information.

Important conceptual point:

- We are not replacing z1.
- z1 is the frozen state sensor.
- z2 should build on z1 and change what cosine/distance means.

Current phrasing:

```text
The whole game is to make the cosines point to different things.
```

Right now z1 cosine points toward board-state / structural similarity. z2 should make cosine/distance point toward:

```text
objective alignment
successor compatibility
terminal-region direction
action-reconstructible transitions
planning-useful state similarity
```

## A/B training bundle idea

A Manifolder training datapoint may consist of 6-8 z1 inputs.

For sampled state `A`:

```text
A
A_next_ply
A_next_ply_same_player
A_terminal
```

For paired state `B`:

```text
B
B_next_ply
B_next_ply_same_player
B_terminal
```

So generally:

```text
[A, A+1, A+2, A_T, B, B+1, B+2, B_T]
```

Potentially fewer if near terminal.

Critical unresolved issue:

```text
How do we pick A and B?
```

This was identified as the core of Manifolder training. The architecture is secondary; the pairing/sampling policy defines the geometry z2 will learn.

Candidate thoughts:

- Ideal B might be z1 nearest neighbor.
- Exact global z1 NN over huge corpora would require large storage:

```text
1M positions  ≈ 24 GB raw fp16 z1
10M positions ≈ 246 GB raw fp16 z1
```

- Existing 100k BSE inspection index can be used for early proof-of-concept pairing.
- More thought is needed before implementing Manifolder trainer.

## Planning manifold principles drafted by user

User drafted a conceptual Section 2: The Planning Manifold with propositions:

1. Terminal states representing different objectives must be distant within the manifold.
   - Distant terminal states confer directionality bias on the vector field.
2. Similar universe states must occupy similar positions on the manifold, for both terminal and non-terminal states.
   - Similarity must be defined by meaningfulness to the final objective, not just action sequence.
3. Successor states aligned with the same objectives must be close, while successor states aligned with different objectives must be distant.
   - Successor states represent evolution; evolution should be comparatively slow, making inverse dynamics easier.
4. Non-terminal distant states aligned with an objective should generally be in the same direction as terminal states aligned with that objective.
   - Needed for generalizing from states similar to but not identical with known good trajectories.

Important correction identified:

The current ticket's old L3 loss is probably wrong / underspecified. It compared consecutive local deltas:

```text
v1 = a_next_ply - a
v2 = a_next_ply_same_player - a_next_ply
1 - cosine_similarity(v1, v2)
```

Problem: adjacent plies in chess can zig-zag tactically and opponent moves may oppose the objective. Smoothness should be objective-conditioned, not merely consecutive-delta similarity.

Better direction:

```text
L_objective_direction or split:
L_successor_locality
L_objective_direction
```

Example formulation:

```text
v_successor = z_next_same_objective - z_current
v_objective = z_terminal_same_objective - z_current
encourage cos(v_successor, v_objective) high
```

and discourage alignment toward opposite-objective successor/terminal regions.

## Next recommended steps

1. Continue using Board State Encoder Inspection to collect qualitative examples.
2. Especially collect cases where:
   - prefix match = no
   - last match = no
   - side to move differs
   - z1 neighbor visually similar
   - Stockfish eval either matches or strongly differs
3. Treat those cases as evidence for what z1 does and does not encode.
4. Think carefully about A/B pairing policy before implementing Manifolder 1000.
5. Update the Scryer Manifolder-tron ticket later with:
   - BSE milestone complete
   - z1/z2 terminology
   - planning manifold propositions
   - corrected L3/objective-direction framing
   - proposed z2 architecture
   - A/B training bundle concept

Do not rush Manifolder code before resolving the sampling/pairing strategy.
