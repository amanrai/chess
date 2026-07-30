# Handoff — 23 Jul

## Big picture

The explicit board-state grounding objective worked extremely well. The current Q-Former board-state encoder/probe is now validated as a queryable latent chess board-state representation, not merely a movetext embedding.

Architecture under test:

```text
move-history prefix -> QFormerPlyHistoryEncoder -> Q-bank [B, K, D]
queried square id -> square embedding -> cross-attend to Q-bank -> occupant logits [B, Q, 13]
```

The square query only receives square id. Board labels are only used in the CE loss. `CHECK` and `MATE` tokens are stripped before encoding to avoid leakage.

## Checkpoints

Local checkpoint directory:

```text
checkpoints/board_state_q_probe/
```

Current local checkpoints:

```text
board_state_q_probe_fast-wildflower-18_epoch_001_batch_050000.pt
board_state_q_probe_fast-wildflower-18_epoch_001_batch_100000.pt
board_state_q_probe_fast-wildflower-18_epoch_001_batch_150000.pt
```

Backup copy exists at:

```text
checkpoints/board_state_q_probe_backup_20260722_235727/
```

Important: held-out eval metrics below were confirmed using the latest available checkpoint:

```text
checkpoints/board_state_q_probe/board_state_q_probe_fast-wildflower-18_epoch_001_batch_150000.pt
```

The 50k checkpoint was loaded earlier only for parameter-count inspection.

Parameter count from the 50k checkpoint/model config:

```text
total parameters:     44,271,635
trainable parameters: 44,271,635
encoder parameters:   31,609,856
board_head:            3,193,869
check_head:            3,155,970
mate_head:             3,155,970
turn_head:             3,155,970
```

Model config:

```text
model_dim: 512
heads: 32
history_layers: 6
q_layers: 4
num_queries: 64
context_plies: 128
squares_per_position: 16
batch_size: 32
grad_accum_steps: 16
```

## Eval scripts added

Committed and pushed:

```text
cc9b92b Add board-state reconstruction eval scripts
e0edabd Clarify board-state eval exact metrics
f15962d Add average wrong squares to board eval
d716c13 Add wrong-square percentiles to board eval
```

New scripts:

```text
scripts/build_board_state_eval_bucket.py
scripts/evaluate_board_state_q_probe.py
```

`build_board_state_eval_bucket.py` samples a held-out Elo bucket and stores:

```text
data/processed/lumbras/eval_board_state_2000_2199/
  games.pgn
  moves.npy
  offsets.npy
  results.npy
  board_after_packed.npy
  manifest.json
```

Command used to build 2000–2199 eval set:

```bash
uv run python scripts/build_board_state_eval_bucket.py \
  --samples 1000 \
  --out-dir data/processed/lumbras/eval_board_state_2000_2199 \
  --seed 20260723
```

Build result:

```text
bucket candidates: 2,362,698
sampled: 1000
accepted after replay/tokenize: 999
total plies: ~79.7k
```

Evaluator command shape:

```bash
uv run python scripts/evaluate_board_state_q_probe.py \
  --eval-dir data/processed/lumbras/eval_board_state_2000_2199 \
  --max-games 100 \
  --batch-size 32 \
  --checkpoint checkpoints/board_state_q_probe/board_state_q_probe_fast-wildflower-18_epoch_001_batch_150000.pt
```

The evaluator defaults to latest checkpoint in `checkpoints/board_state_q_probe` if `--checkpoint` is omitted, and uses CUDA automatically if available.

## Held-out board reconstruction results

Dataset: held-out Lumbras OTB games where both players' minimum Elo is in `[2000, 2200)`, below the 2200+ training cutoff.

Eval: 100 sampled held-out games / 8,072 post-ply positions, all 64 squares queried per position, latest 150k checkpoint.

Overall:

```text
exact_square: 0.9907
exact_board:  0.6381
avg_wrong_sq: 0.598
p50_wrong:    0
p75_wrong:    1
p90_wrong:    2
occ_p:        0.9895
occ_r:        0.9845
occ_n:        190,393
pred_occ_n:   189,437
```

Definitions:

- `exact_square`: per-square occupant accuracy over all queried squares.
- `exact_board`: all 64 squares match for a position; one wrong square makes the board non-exact.
- `avg_wrong_sq`: average number of wrong squares per reconstructed board.
- `p50/p75/p90_wrong`: percentiles of wrong-square count per board.

Interpretation:

```text
median reconstructed board is exact
75% of positions have <=1 wrong square
90% of positions have <=2 wrong squares
average board has <0.6 wrong squares
```

Selected 5-ply bucket behavior:

```text
1-25 plies:   p50=0, p75=0, p90=0 across each 5-ply bucket
26-35 plies:  p50=0, p75=0, p90=1
36-40 plies:  p50=0, p75=1, p90=1
41-45 plies:  p50=0, p75=1, p90=2
46-50 plies:  p50=1, p75=1, p90=2
51-55 plies:  p50=1, p75=2, p90=2
56-85 plies:  p50=1, p75=2, p90=3
91-95 plies:  p50=1, p75=1, p90=2
96-105 plies: p50=0, p75=1, p90=2-3
106-125:      mostly p50=1, p75=1-2, p90=2-3
126-135:      small-N tail; p90=4-5
```

Hardest well-sampled region is the tactical/midgame band around roughly 51–85 plies, but even there the model is usually within 1–3 squares of the exact board.

Per-piece metrics from the same eval:

```text
EMPTY         precision 0.9914 recall 0.9943
WHITE_PAWN    precision 0.9949 recall 0.9951
BLACK_PAWN    precision 0.9975 recall 0.9958
WHITE_KING    precision 0.9913 recall 0.9978
BLACK_KING    precision 0.9920 recall 0.9970
WHITE_BISHOP  precision 0.9929 recall 0.9915
BLACK_BISHOP  precision 0.9921 recall 0.9833
WHITE_QUEEN   precision 0.9883 recall 0.9866
BLACK_QUEEN   precision 0.9929 recall 0.9817
WHITE_KNIGHT  precision 0.9793 recall 0.9658
BLACK_KNIGHT  precision 0.9839 recall 0.9581
WHITE_ROOK    precision 0.9643 recall 0.9494
BLACK_ROOK    precision 0.9647 recall 0.9476
```

Conclusion: BSE/Q-bank is a strong queryable current-board representation.

## PM / Scryer tickets updated

Gateway:

```text
http://100.105.192.98:43210/
```

Board-state ticket:

```text
Title: Implement board-state encoder with probe backend
ID: ebd61b46-7493-433e-b7b4-b411430f8e9c
Project: Chess
```

Updated with implementation details, checkpoint clarification, held-out metrics, p50/p75/p90 wrong-square stats, and interpretation.

New ticket created:

```text
Title: Manifold separator
ID: bb06ffbb-ae49-4616-9e19-8f9d2d01a90e
Project: Chess
Status: unopened
Tags: modeling, manifold, board-state, planning, feature
```

## Manifold separator decisions so far

The manifold separator is intended to train geometry of terminal outcome states, not generic partial-position prediction.

### Motivation

Raw BSE/Q-probe space represents board states from which multiple outcomes may be possible. For diffusion/vector-field planning, the same current state should lead to different vector fields depending on desired outcome:

```text
same current q_bank + desired WHITE_WIN -> one destination/vector field
same current q_bank + desired BLACK_WIN -> different destination/vector field
```

If terminal win/draw/loss manifolds are not separated first, later vector fields may point ambiguously toward correct and incorrect win states.

### Data unit

Important correction: v1 uses final complete-game states only, not all intermediate prefixes.

```text
one complete game
  -> full move history through game end
  -> frozen BSE
  -> final q_bank [64, 512]
  -> final result label
```

Labels:

```text
WHITE_WIN
DRAW
BLACK_WIN
```

Absolute labels, not side-to-move-relative. Draw is its own meaningful region: better than loss, not equivalent to win.

Corpus for v1:

```text
current 2200+ verifier corpus
~2.91M final game states
```

### BSE usage for separator

Initially considered offline q_bank cache, but full float32 cache would be huge:

```text
2.91M * 64 * 512 * 4 bytes ~= 381 GB decimal / 355 GiB
```

Decision: v1 should use online frozen BSE, not mandatory latent cache.

```text
final move-history prefix -> frozen BSE -> q_bank -> separator
```

BSE is frozen. Only separator modules train.

Checkpoint policy:

```text
use latest available board-state checkpoint by default
print resolved checkpoint path loudly
allow explicit --checkpoint override
```

Expected latest checkpoint currently:

```text
checkpoints/board_state_q_probe/board_state_q_probe_fast-wildflower-18_epoch_001_batch_150000.pt
```

Optional future optimization: add q_bank cache if online BSE throughput bottlenecks. If cache is added, dtype should be configurable; float16 default for storage efficiency, but float32 first serious run to avoid dtype changes.

### Separator structure

Two modules:

```text
Z: q_bank [64, d] -> z_bank [64, d_sep]
M: z_bank -> absolute result logits [3]
```

Defaults/constraints:

```text
d_sep configurable, default q_bank dim = 512
Z must preserve bank shape, not collapse to a single vector
M can be DiffThinkerMLP-style readout with num_outputs=3
Z architecture still TBD
```

Why `z_bank`, not pooled vector:

Downstream planner likely consumes:

```text
(current q_bank, diffused/target z_bank) -> q_{t+1}-like next state
(q_t, q_{t+1}) -> next move / legal successor selection
```

So separator output should remain bank-shaped.

### Curriculum

Phase 0 — classifier grounding:

```text
q_bank_final -> Z -> z_bank -> M -> logits[3]
loss: CE(WHITE_WIN / DRAW / BLACK_WIN)
train: Z + M
sampling: natural distribution initially
```

Purpose:

```text
make M coherent as an absolute-result predictor
make z_bank outcome-aware before triplet shaping
```

Phase 1 — triplet manifold shaping:

```text
anchor_final_q, positive_same_result_q, negative_different_result_q
  -> shared Z -> z_bank
  -> shared M -> logits[3]
```

Train both `Z` and `M` in phase 1. CE remains necessary to keep `M` coherent while triplet loss shapes `Z`.

Loss sketch:

```text
CE(anchor, y_anchor) + CE(positive, y_positive) + CE(negative, y_negative)
+ lambda * cosine_triplet(z_anchor, z_positive, z_negative)
```

Triplet construction:

```text
positive = same final result as anchor
negative = any different final result
```

Do not hard-code ordinal fallback into triplets. Later planner/search should decide whether to settle for draw if win is not achievable.

### Open separator questions

Still undecided:

- exact `Z` architecture;
- exact cosine distance reduction over `z_bank [64, d]`;
- `Z` output normalization/activation;
- phase 0 natural vs balanced sampling if natural fails;
- whether/when to add optional q_bank caching.

## Z output geometry discussion

We discussed `tanh` vs L2 normalization.

`tanh`:

```text
z_i = tanh(alpha * x_i)
```

- coordinate-wise bounded box `[-1, 1]`;
- alpha/slope controls saturation;
- potentially useful for future diffusion target coordinates;
- can saturate and reduce gradients.

L2 normalization:

```text
z = x / ||x||
```

- vector-wise unit-sphere geometry;
- common for contrastive/triplet/cosine metric learning;
- avoids per-coordinate tanh saturation and norm cheating;
- if diffusion generates z later, it may need to stay on/near the sphere via normalize/project steps.

Practical simplification: if using L2/cosine, generated states can be kept as directions by normalizing after updates:

```python
z_next = normalize(z + delta)
```

No decision yet; worth thinking about because this affects gradient shape and downstream diffusion geometry.

## Go analogy

This approach likely transfers naturally to Go:

```text
SGF/move history -> latent state bank -> coordinate query -> point state
```

Go board state is wider (`19x19 = 361` points), but the same queryable latent-state architecture should apply. Future outcome manifold/vector-field ideas may be even more natural in Go because strategic influence is highly geometric.

## Critical doubts / things to keep rethinking

These are not side notes. They are load-bearing conceptual questions. BSE supports the manifold separator; the diffuser/vector-field work will build on the separator; the thinker/state-transitioner/move-generator will build on all of them. If any geometry choice is wrong, downstream modules may learn the wrong thing very efficiently.

### 1. What exactly is the separator separating?

Current decision: train on final complete-game states only.

Reason:

```text
we are trying to learn terminal/win-state geometry, not early-position outcome prediction
```

Doubt to revisit:

```text
final positions may encode obvious terminal artifacts rather than broadly useful win manifolds
```

For example, final states may be checkmates, resignable positions, drawn endings, etc. We strip CHECK/MATE tokens from the BSE input, but final board states can still be structurally extreme. This may still be okay because the goal is destination manifolds, but it is worth remembering.

### 2. Are absolute labels sufficient?

Current decision:

```text
WHITE_WIN / DRAW / BLACK_WIN
```

Not side-to-move-relative.

Reason:

```text
we want outcome-conditioned destination regions; side-to-move can be derived separately
```

Doubt to revisit:

```text
planning may eventually need player-relative geometry: current-player-win / draw / loss
```

Absolute manifolds are probably right for terminal state separation. But move selection may need to interpret them through side-to-move. Do not forget this when designing the planner.

### 3. Draw geometry is ambiguous

Current decision: draw is its own third manifold.

Reason:

```text
draw is not a win, but it is better than loss
```

Doubt:

```text
should draw be geometrically between white-win and black-win, or a separate region entirely?
```

We explicitly decided not to hard-code ordinal fallback into triplets. But later planning will need to know when to settle for draw if win is not achievable. This may require an additional value/preference layer on top of separated manifolds.

### 4. Natural distribution vs balanced classes

Current decision for phase 0:

```text
natural result distribution initially
```

Reason:

```text
first see whether the signal works without intervention
```

Doubt:

```text
if draws dominate or one result class dominates, M may become a strong classifier mostly by priors
```

If phase 0 looks too prior-driven, add balanced sampling or class-weighted CE. But do not do that before seeing the baseline.

### 5. Online frozen BSE vs latent cache

Current decision:

```text
online frozen BSE for v1
```

Reason:

```text
full q_bank cache for 2.91M games is ~381 GB in float32
```

Doubts:

```text
online BSE may bottleneck separator training throughput
using BSE online means training cost is dominated by a frozen network
cache may become necessary if iterations are too slow
```

If caching is added, dtype must be considered carefully. Float16 would save disk, but the first serious run may want float32 to avoid changing representation geometry midstream.

### 6. Context limit for final states

BSE context is currently:

```text
context_plies = 128
```

So final complete-game state actually means:

```text
last 128 plies of the game, padded if shorter
```

Doubt:

```text
for games longer than 128 plies, the BSE final state does not see the full game history
```

The BSE reconstruction eval still worked well deep into games, but terminal separator training should remember this limit. Long-game final q_banks are compressed/context-truncated final states.

### 7. Z architecture is unresolved and important

Current constraints:

```text
Z: q_bank [64, d] -> z_bank [64, d_sep]
d_sep defaults to d = 512
Z must preserve bank structure
```

Doubt:

```text
how much should Z be allowed to rewrite the BSE state?
```

Too weak: manifolds may not separate.

Too strong: Z may invent a geometry M likes but that later diffuser/planner cannot use as a meaningful state target.

Possible Z families to revisit:

```text
per-slot residual MLP
small self-attention stack over Q slots
residual transformer blocks
gated residual update
LayerNorm + projection + activation
```

### 8. M architecture is easier but still matters

Current inclination:

```text
M = DiffThinkerMLP-style readout, num_outputs=3
```

Doubt:

```text
if M is too strong, it may classify outcomes while Z geometry remains weak
```

This is especially relevant in phase 0. Phase 1 triplet loss is supposed to force Z geometry, but if M can do too much, classifier accuracy alone will be misleading.

### 9. Cosine triplet distance over z_bank is not fully specified

Current phase 1 idea:

```text
cosine_triplet(z_anchor, z_positive, z_negative)
```

Doubt:

```text
what is cosine distance between two [64, d] banks?
```

Options:

```text
flatten whole bank and compute cosine once
compute per-slot cosine and average
learn slot attention/weights for distance
use pooled z only for distance but keep bank output
```

This choice affects what kind of geometry the diffuser later sees.

### 10. tanh vs L2 normalization is unresolved

`tanh` option:

```text
z_i = tanh(alpha * x_i)
```

Pros:

```text
bounded coordinate box [-1, 1]
possibly easier for diffusion target generation
alpha controls slope/saturation
```

Cons:

```text
coordinate-wise saturation can kill gradients
alpha tuning becomes another knob
may push coordinates to extremes
```

L2 normalization option:

```text
z = x / ||x||
```

Pros:

```text
standard for cosine/triplet metric learning
prevents norm cheating
no tanh saturation
clean angular geometry
```

Cons:

```text
future diffusion may need to occur on/near sphere
generated states may need normalize/project steps
sphere geometry may complicate intuition/training
```

Important conceptual question:

```text
if z lives on a sphere, should diffusion/vector fields move along the sphere?
```

Practical approximation:

```python
z_next = normalize(z + delta)
```

More geometric approximation:

```python
v = delta - dot(delta, z) * z   # tangent-ish projection
z_next = normalize(z + eta * v)
```

No decision yet. This is worth rethinking before implementing Z/output activation.

### 11. Phase 1 must train M too

Initial thought was maybe freeze M after phase 0 and train only Z with triplets.

Updated thought:

```text
train Z + M in phase 1
keep CE on all triplet entries
```

Reason:

```text
M must remain coherent as Z geometry changes
```

Doubt:

```text
CE and triplet loss may fight each other depending on Z output geometry
```

Need to monitor both classifier accuracy and geometry metrics.

### 12. Triplet construction intentionally avoids ordinal preferences

Current decision:

```text
positive = same final result
negative = any different result
```

Reason:

```text
whether to settle for draw if win is not available should be algorithmic/planning-time logic, not baked into separator triplets
```

Doubt:

```text
without ordinal structure, draw may not sit in a useful fallback relationship to win/loss
```

This may be okay because fallback belongs to planner/value layer, but remember to revisit.

### 13. How to know separator worked?

Phase 0 CE accuracy is not enough.

Need metrics like:

```text
3-way classifier accuracy / confusion matrix
per-class precision/recall, especially draws
intra-class vs inter-class z distances
triplet violation rate
nearest-neighbor outcome purity in z space
embedding projections/visualizations
whether same-result states cluster by outcome rather than superficial material/terminal artifacts
```

Potential risk:

```text
separator learns terminal-material heuristics, not useful outcome manifolds
```

This may still be useful, but downstream planning might expose the difference.

### 14. Budget / proof strategy

Rough current spend on first BSE run: ~$50.

Willingness: spend another ~$100 to prove the whole stack at small scale:

```text
manifold separator
diffuser/vector field
thinker/future-state generator
move generator / legal successor selector
```

Then, if the system works conceptually, spend another ~$150-ish to rebuild correctly with better scaling/cleaner runs.

Implication:

```text
v1 should prove the point, not optimize everything
```

But geometry decisions should still be documented carefully because the entire stack builds on them.

## Repo status

Scripts are checked in and pushed. Current untracked local file should only be:

```text
uv.lock
```

`handoff_21_jul.md` was deleted and replaced by this file.
