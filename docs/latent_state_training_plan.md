# Latent State Training Plan

## Core hypothesis

The Q-former verifier run is a proof-of-life for the architecture: a compact latent state produced from chess movetext can contain enough information to predict the eventual winner above random levels. That means the encoder is not just memorizing syntax; it is extracting outcome-relevant state information.

The next question is not whether the latent state can be measured. It can. The next question is how to train a module that predicts a useful next or future good state.

The long-term goal is:

```text
current state -> latent state -> predicted good future state -> choose move that moves toward it
```

This is different from direct next-move imitation. The model should first imagine what a better future state should look like, then derive a move from the difference between the current state and that desired future state.

## Current evidence

### Full-game verifier

Training with full-game samples already shows strong early signal. If the verifier can predict the winner well above random from the latent state, then the encoder/value head can measure state quality.

This validates:

- the move-history encoder
- the Q-former latent bottleneck
- the result/value prediction head
- the idea that the latent state carries meaningful chess information

### Prefix verifier

The prefix run probes a harder question: how much outcome and trajectory information is already available before the game is complete?

With prefix sampling, the model sees incomplete games and predicts the final result. If it learns well from prefixes, especially around half-game contexts, then the state already contains enough signal to guide future-state prediction. If it does not, future phases may need more elaborate training signals.

## Phase 1: state/value pretraining

Train the shared Q-encoder with a result prediction objective.

Inputs:

- full games
- late-game contexts
- possibly mixed prefixes later

Targets:

- final result: white win, black win, draw

Purpose:

- make the latent state outcome-aware
- teach the encoder to compress game history into a measurable state
- produce a reusable value/verifier head

Initial command shape:

```bash
uv run python scripts/train_encoder_q.py \
  --batch-size 8 \
  --sample-mode full \
  --max-game-moves 200 \
  --model-dim 256 \
  --heads 16 \
  --grad-accum-steps 16
```

Prefix probe command shape:

```bash
uv run python scripts/train_encoder_q.py \
  --batch-size 32 \
  --sample-mode prefix \
  --prefix-fraction 0.5 \
  --context-moves 125 \
  --min-game-moves 100 \
  --max-game-moves 250 \
  --model-dim 256 \
  --heads 16 \
  --grad-accum-steps 16
```

For interpretation, prefix experiments should prefer percentage-based prefixes over fixed context-only controls. `--context-moves` is only the model window; it does not define how much of the original game is sampled. A half-game probe should explicitly set `--prefix-fraction 0.5`, then set `--context-moves` large enough to avoid truncating the sampled half-game prefix.

## Phase 2: contrastive trajectory training

Use a Siamese/shared-encoder architecture to teach the latent space about which future states belong to the same trajectory.

Architecture:

```text
z_t   = encoder(prefix/current state)
z_pos = encoder(future state from the same game)
z_neg = encoder(future state from another game)
```

Training objective:

```text
score(z_t, z_pos) > score(z_t, z_neg)
```

Equivalent losses to consider:

- InfoNCE / contrastive cross entropy
- triplet loss
- margin ranking loss
- binary same-game classifier over pairs

Purpose:

- make latent distances meaningful
- teach the model which futures are compatible with the current state
- preserve trajectory information beyond simple result prediction
- prepare the latent space for generative future-state modeling

Negative sampling curriculum:

1. random game negatives
2. same result negatives
3. same opening-family negatives
4. same phase/length negatives
5. same material-ish or similar-position negatives, if board features are added
6. same player/event negatives, if metadata is available

Training policy:

- start from the Phase 1 encoder
- initially freeze or mostly freeze the encoder
- train the projection/scoring heads
- then lightly unfreeze the encoder
- avoid destroying the outcome-aware structure learned in Phase 1

## Phase 3: latent future-state generation

Train a generative model over future latent states. The likely direction is a diffusion/denoising process in latent space.

Conceptual model:

```text
noisy future latent + current latent + side-to-move -> denoised plausible future latent
```

The model should learn to generate plausible future states conditioned on the current state. Then the Phase 1 verifier/value head can steer selection toward high-value generated states.

Possible objective:

```text
predict noise / denoise z_future given z_current and timestep
```

Conditioning signals:

- current latent state
- side to move
- move number / phase
- desired result or value direction
- optional value guidance from the verifier

Purpose:

- generate a desired future state rather than directly generate a move
- allow strategic planning in latent space
- use the verifier as a quality signal over imagined futures

## Move derivation from desired future state

The final system should choose legal moves by comparing their resulting latent states to the desired future latent.

Basic selection:

```text
for each legal move m:
    apply m to current board/history
    encode resulting state -> z_m
choose m that best matches z_desired
```

Scoring candidates:

```text
score(m) = similarity(z_m, z_desired) + value(z_m)
```

Possible terms:

- cosine similarity to generated future state
- verifier/value score of candidate state
- legal move prior, if trained
- tactical safety checks from an engine or rule-based filter during development

This keeps the generator focused on predicting good states, while the move chooser handles legal action selection.

## If prefix training is strong

If the prefix verifier learns meaningful signal from partial games, then the path can stay relatively simple:

1. keep training the state/value encoder
2. add contrastive same-trajectory learning
3. train latent transition or latent diffusion over future states
4. derive moves by matching legal successor states to desired futures

This would suggest the latent already captures enough information about game trajectory and position quality.

## If prefix training is weak

If prefix training does not climb much above baseline, then incomplete movetext alone may not provide enough supervision. Add auxiliary objectives.

Candidates:

- board/FEN reconstruction from latent state
- side-to-move prediction
- legal move prediction
- material balance prediction
- check/castling/en-passant status prediction
- engine evaluation targets for intermediate positions
- contrastive good-continuation vs bad-continuation pairs

The goal would be to enrich the latent state before asking it to support future-state generation.

## Evaluation plan

Eventually add explicit eval loops. For now training metrics are enough for rough signal, but future evaluation should split results by prefix bucket and data partition.

Important evaluations:

- held-out games
- held-out players/events if metadata exists
- prefix bucket accuracy:
  - first 20%
  - 20-50%
  - 50-80%
  - final 20%
- result-class precision/recall
- baseline comparison against class priors
- opening-family stratification

Interpretation:

- weak first 20% but strong later buckets means the model is learning game trajectory and position quality
- strong first 20% may include opening/population priors and should be interpreted carefully
- robust performance on held-out players/events would be much stronger evidence of chess understanding

## Open design questions

- Should Phase 2 predict actual future states, better future states, or both?
- Should the future target be next ply, next 8 plies, next 32 plies, or a random later prefix?
- Should draws be modeled as a separate strategic target or downweighted initially?
- Should the encoder latent be frozen during diffusion training?
- How should generated future states be constrained to be reachable?
- How much board-state supervision is needed beyond movetext?

## Working thesis

The best version may be a staged system:

```text
Phase 1: value-aware state encoder from full games
Phase 2: contrastive trajectory learning with shared encoder
Phase 3: latent diffusion/generation of plausible good futures
Phase 4: legal move selection by matching successor states to generated futures
```

The full-game verifier result suggests Phase 1 is already viable. The current prefix experiments determine whether Phase 2 and Phase 3 can be simple latent-space extensions or need richer auxiliary supervision.
