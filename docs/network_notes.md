# Network notes

Working notes while discussing architecture. Not final.

## Core idea

- Combine two ideas: TRM + diffusion.

## TRM part

- Use variable widths.
- Conceptually split model into `thinker` and `generator`.
- `thinker` is substantially wider than `generator`.
- Token embedding may be even wider than `thinker`.
- Variation: introduce a deliberate bottleneck within `thinker` to test the hypothesis that bottlenecks lead to better generalization.
- `thinker` receives:
  - previous board state
  - planned move
- First iteration planned move can be all `<PAD>`.
- `thinker` output goes to `generator`.
- `generator` attempts to generate the next move.
- Generate/refine the next move for `k` iterations.
- Ask `thinker` to keep improving the planned move during those iterations.
- Feed the previously chosen/planned move back into the `z_L` and `z_H` networks.
- Evolve `z_L` / `z_H` state iteratively.
- During training, take the greedy output and refeed that.

## Diffusion part

- Predict all tokens in the ply packet in a single shot.
- Treat ply packet tokens like masked tokens to be filled jointly.
- During inference, may explore entropy-based improvement / refinement.

## Board state / Q-network-ish idea

- Open question: do we need a separate board-state embedding?
- It may be useful, but not committed yet.
- Board representation could be 64 squares with piece-type/color features.
- Approx framing: `thinker(existing board state, projected move) -> new board state`.
- This starts to look like a Q-network / transition model.
- One possible path: thinker evolves into a policy network.
- This part is still unsettled.
- Move history stream is useful, but it is not more representative of state than current board state.
- If `thinker` is expected to simulate/search from a position, the Markov assumption should mostly stand: current board state should be sufficient for legal continuation.
- Exceptions / non-board state needed for full chess state:
  - side to move
  - castling rights
  - en passant square
  - halfmove clock / repetition context if drawing rules matter
  - maybe move number / phase as derived context
- Move history may still be useful as style/opening/player-distribution context, but not as the primary state representation.
- Open question to investigate: pure ply-history AR model vs explicit board-state model.
- Notebook next step: implement a basic pure autoregressive transformer first.
- No thinker yet.
- No board-state embedding yet.
- Main special thing: RoPE position increments by ply index, not by token slot.

## Thinker / board encoder direction

- Likely need an explicit board-state encoder for the thinker.
- Minimal board input:
  - `board_state`: 64 square piece ids
  - extra chess state: side to move, castling rights, en-passant square, maybe halfmove/repetition info
- Minimal encoder:
  - piece embedding per square
  - square position embedding
  - small transformer or MLP over 64 squares
  - pooled `board_emb`
- Planned ply packet gets its own move encoder.
- Possible thinker inputs:
  - `board_emb`
  - planned move embedding
  - optional ply-history/context embedding
  - recurrent/iterative `z_L`, `z_H` state
- Iterative loop:
  - `plan_0 = <PAD>` packet
  - for `t in 0..k-1`:
    - `thought_t = thinker(board_emb, plan_t, z_L, z_H)`
    - `logits_t = generator(thought_t)`
    - `plan_{t+1} = greedy(logits_t)`

## Generator variants

- Add an AR generator variant to round out the test bed.
- Generator variant A: diffusion / masked packet generator.
  - Predicts all 8 move slots jointly.
- Generator variant B: AR packet generator.
  - Predicts move slots left-to-right within the ply packet.
  - Example order: piece/source/capture/destination/promotion/check-or-mate/`<EOM>`.
- Keep thinker mostly constant and swap generator type.
- This gives a clean ablation: joint move prediction vs intra-move autoregression.

## Ablations / hypotheses

### Generator ablation

- Joint packet generator vs AR packet generator.
- Question: is it better to predict the whole ply packet at once or autoregress within the move?

### Verifier prefix sampling

- Verifier training should distinguish clearly between:
  - prefix length: how much of the original game is sampled
  - context window: how many sampled ply packets the model can see after crop/pad
- For controlled partial-game experiments, prefer percentage-based prefixes such as exactly 50% of each game.
- Fixed context alone is not enough for interpretation: `context_plies=100` can include entire short games, which makes a supposed half-game experiment ambiguous.
- Same game can provide many samples by choosing different prefix lengths.
- Prefix length distribution matters:
  - early prefixes: opening/result priors, high uncertainty
  - mid prefixes: positional evaluation
  - late prefixes: conversion/decisive-state recognition
- Support exact fraction prefixes, random fraction ranges, absolute ply buckets, and fraction-of-game buckets.
- Training sampler should choose: game passing length filters -> prefix fraction or bucket -> prefix length -> crop/pad context.
- Preprocessing can be iterated on a Tailnet/Jupyter machine before transferring compact tokenized datasets to Vast.

### Encoder fact probes

Use simple auxiliary probe tasks to separate encoder failure from WDL-label ambiguity.

Initial probes:

- predict whether the final sampled prefix ply resulted in check/mate
- predict whose turn is next after the sampled prefix

For the check/mate probe, remove `CHECK` and `MATE` tokens from the input and use them only as labels to avoid leakage.

### State representation ablation

- Pure ply-history AR model.
- Explicit board-state encoder model.
- Board-state + ply-history hybrid.
- Question: does explicit Markov state reduce required context / improve legality / improve strength?

### Thinker/generator ablation

- No thinker baseline.
- Thinker + generator iterative model.
- Question: does iterative thinking improve move quality at fixed parameter count?

### Scaling hypothesis

- Scale thinker width/depth.
- Scale generator width/depth.
- Scale token embedding width.
- Scale context length.
- Scale number of refinement iterations `k`.
- Scale bottleneck size.
- Scale dataset size / Elo threshold.
- Main question: does scaling thinker capacity matter more than scaling generator capacity?
- Other questions:
  - Does larger `k` substitute for more parameters?
  - Does bottlenecking improve generalization at fixed compute?
  - Does board-state encoding reduce the amount of ply history needed?

### Bottleneck hypothesis

- Add deliberate bottleneck inside thinker.
- Compare against same-param or same-compute non-bottleneck thinker.
- Question: do bottlenecks improve generalization?

## Positional encoding / RoPE

- Each move is represented by multiple tokens: one packet of up to 8 token slots.
- The collection of tokens represents a single move.
- RoPE must be applied equally to all tokens within a ply packet.
- RoPE position increments by **ply index**, not by token slot.
- Tensor is conceptually `[batch, move_index, slot_index]`, not just a flat token stream.
