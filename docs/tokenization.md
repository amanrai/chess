# Chess movetext tokenization design

This document records the current tokenization choices for the chess model project. The goal is to make the notation representation explicit before we tokenize the full Lumbras dataset or design the network in detail.

## Status

This is an early design checkpoint, not a frozen data format.

We have implemented and tested a reusable tokenizer/detokenizer in:

- `src/chessgm/tokenizer.py`
- `tests/test_tokenizer.py`
- `scripts/tokenize_move.py`
- `notebooks/chess_tokenizer.ipynb`

We have **not** run full dataset tokenization yet. That is intentional: the network design may change how examples are packed, how context is represented, and whether the output head predicts a move packet, a flattened sequence, or something more structured.

## What is being tokenized

We tokenize **PGN movetext / SAN moves**, not metadata.

PGN metadata lines such as:

```pgn
[White "Carlsen, Magnus"]
[Black "Nakamura, Hikaru"]
[WhiteElo "2830"]
```

are not part of the core move-token stream.

The current tokenizer is a **notation codec**:

```text
SAN move <-> structured tokens <-> integer ids
```

It is not a board-state simulator and it does not validate move legality. Legal move validation, legal action masking, and environment stepping should be handled later by an external chess environment, likely `python-chess` initially.

## Why not one token per full move?

A full SAN move vocabulary is possible, but it bakes too much structure into opaque strings.

Examples of opaque full-move tokens would be:

```text
Nbd7
Qxf7+
exd8=Q#
O-O
```

That approach loses useful internal structure:

- which piece moved
- whether the move captured
- destination square
- promotion piece
- check / mate status
- source disambiguation

Instead, we use compositional move tokens. This keeps the vocabulary small while preserving chess-relevant structure.

## Core representation

Each SAN move becomes a variable-length **move packet** ending in `<EOM>`.

For model convenience, packets are currently padded to a fixed length:

```text
MOVE_SEQ_LEN = 8
```

Examples:

```text
e4        -> PAWN TO_e4 <EOM> <PAD> <PAD> <PAD> <PAD> <PAD>
Nf3       -> PIECE_N TO_f3 <EOM> <PAD> <PAD> <PAD> <PAD> <PAD>
Nbd7      -> PIECE_N SRC_FILE_b TO_d7 <EOM> <PAD> <PAD> <PAD> <PAD>
R1e2      -> PIECE_R SRC_RANK_1 TO_e2 <EOM> <PAD> <PAD> <PAD> <PAD>
Bxc6      -> PIECE_B CAPTURE TO_c6 <EOM> <PAD> <PAD> <PAD> <PAD>
dxc6      -> PAWN SRC_FILE_d CAPTURE TO_c6 <EOM> <PAD> <PAD> <PAD>
exd8=Q+   -> PAWN SRC_FILE_e CAPTURE TO_d8 PROMO_Q CHECK <EOM> <PAD>
O-O       -> CASTLE_KINGSIDE <EOM> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>
O-O-O#    -> CASTLE_QUEENSIDE MATE <EOM> <PAD> <PAD> <PAD> <PAD> <PAD>
```

A full game can then be represented as a sequence of move packets:

```text
[num_moves, 8]
```

For batched training examples:

```text
x: [batch, context_moves, 8]
y: [batch, 8]
```

where `x` is a context window of previous move packets and `y` is the next move packet.

## Token categories

### Control tokens

Control tokens use angle brackets so they are visually and programmatically distinct from chess notation tokens.

```text
<PAD>
<BOS>
<EOS>
<GAME_START>
<GAME_END>
<UNK>
<EOM>
```

Important current ids:

```text
<PAD> = 0
<EOM> = 6
```

`<PAD>` fills unused slots in a fixed-width move packet.

`<EOM>` marks the end of the real move. This is important because move packets are variable-length semantically but fixed-length physically.

### Turn tokens

```text
WHITE
BLACK
```

These are optional in movetext tokenization. For single-move tokenization they are not included by default. For sequence examples they can be useful, especially if we later slice games at arbitrary points or represent incomplete context windows.

### Piece / actor tokens

```text
PAWN
PIECE_K
PIECE_Q
PIECE_R
PIECE_B
PIECE_N
```

SAN omits the pawn letter, but the tokenizer emits `PAWN` explicitly.

### Source disambiguation tokens

```text
SRC_FILE_a ... SRC_FILE_h
SRC_RANK_1 ... SRC_RANK_8
```

These cover SAN disambiguation:

```text
Nbd7 -> PIECE_N SRC_FILE_b TO_d7 <EOM>
R1e2 -> PIECE_R SRC_RANK_1 TO_e2 <EOM>
```

For pawn captures, the source file is also represented with `SRC_FILE_*`:

```text
exd5 -> PAWN SRC_FILE_e CAPTURE TO_d5 <EOM>
```

### Destination square tokens

```text
TO_a1 ... TO_h8
```

There are 64 destination tokens.

### Action / tactical suffix tokens

```text
CAPTURE
CHECK
MATE
```

Examples:

```text
Qxf7+ -> PIECE_Q CAPTURE TO_f7 CHECK <EOM>
Qh7#  -> PIECE_Q TO_h7 MATE <EOM>
```

### Castling tokens

```text
CASTLE_KINGSIDE
CASTLE_QUEENSIDE
```

Examples:

```text
O-O    -> CASTLE_KINGSIDE <EOM>
O-O-O  -> CASTLE_QUEENSIDE <EOM>
```

Check/mate suffixes still apply:

```text
O-O-O# -> CASTLE_QUEENSIDE MATE <EOM>
```

### Promotion tokens

```text
PROMO_Q
PROMO_R
PROMO_B
PROMO_N
```

Example:

```text
exd8=Q+ -> PAWN SRC_FILE_e CAPTURE TO_d8 PROMO_Q CHECK <EOM>
```

### Result tokens

```text
RESULT_WHITE   # 1-0
RESULT_BLACK   # 0-1
RESULT_DRAW    # 1/2-1/2
RESULT_UNKNOWN # *
```

Result tokens terminate a game. They do not receive `<EOM>`, because they are not moves.

## Vocabulary size

The current vocabulary size is 108 tokens.

Approximate breakdown:

```text
7 control tokens
2 turn tokens
6 piece / actor tokens
8 source-file tokens
8 source-rank tokens
64 destination-square tokens
3 action tokens
2 castle tokens
4 promotion tokens
4 result tokens
```

A 150-token budget is still comfortably above the current design and leaves room for future additions.

Possible future additions:

- explicit `MOVE_START`
- derived phase tokens
- clock/time-control tokens if we ever keep clock data
- annotation/NAG tokens if annotations become useful
- engine-evaluation bucket tokens if we add labels
- special masking tokens for diffusion/noising schemes

## Integer ids

The tokenizer exposes stable maps:

```python
from chessgm.tokenizer import ChessTokenizer

tok = ChessTokenizer()
tok.token_to_id
tok.id_to_token
```

Example:

```python
encoded = tok.encode_move("Nbd7")
encoded.tokens
# ['PIECE_N', 'SRC_FILE_b', 'TO_d7', '<EOM>']

encoded.ids
# [14, 16, 82, 6]

tok.decode_move(encoded.ids)
# 'Nbd7'
```

The command-line helper pads to a fixed move length of 8 by default:

```bash
./scripts/tokenize_move.py Nbd7
```

Output:

```text
tokens: ['PIECE_N', 'SRC_FILE_b', 'TO_d7', '<EOM>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
ids:    [14, 16, 82, 6, 0, 0, 0, 0]
```

## Detokenization

The tokenizer supports round-tripping:

```text
SAN -> tokens -> ids -> tokens -> SAN
```

The detokenizer reconstructs canonical SAN from a single move packet. It ignores:

- `<PAD>`
- `<EOM>`
- optional `WHITE` / `BLACK`
- other structural boundary tokens where appropriate

Examples:

```text
[PIECE_N, SRC_FILE_b, TO_d7, <EOM>, <PAD>, ...] -> Nbd7
[PAWN, SRC_FILE_e, CAPTURE, TO_d8, PROMO_Q, CHECK, <EOM>, <PAD>] -> exd8=Q+
```

## How this might feed a network

The current leading idea is that the model predicts a whole next-move packet, not a single monolithic move id.

A simple training item could be:

```text
x = previous K move packets: [K, 8]
y = next move packet:        [8]
```

Batch shape:

```text
x: [batch, context_moves, 8]
y: [batch, 8]
```

For an autoregressive model, the 8 packet slots could be predicted left-to-right.

For a diffusion-style output head, the 8 packet slots could be denoised jointly. Each slot is a categorical variable over the vocabulary. `<PAD>` and `<EOM>` provide explicit structure so the head can learn shorter versus longer moves.

Important distinction:

- `<PAD>` is a structural padding target, not a legal chess token.
- `<EOM>` is a structural boundary token, not a legal chess token.
- Legality still requires a board-state environment.

## Legal move validation is separate

This representation can express legal and illegal moves. For example, it can express:

```text
PIECE_Q TO_h5 <EOM>
```

without knowing whether a queen can actually move to h5 from the current board.

During evaluation or self-play, we will need an external environment layer:

```text
board state + candidate decoded SAN/packet -> legal/illegal
```

Likely initial responsibilities for that environment:

- maintain board state
- generate legal moves
- parse SAN against the current board
- map legal moves into token packets
- mask invalid model outputs or reject/resample them
- push accepted moves onto the board

The tokenizer should remain a compact notation layer, not a full chess rules engine.

## Why full dataset tokenization is deferred

Although `scripts/tokenize_pgn_dataset.py` exists and has been smoke-tested, we should not treat its output format as final yet.

Open design questions for the network may affect dataset packing:

1. Are examples move-level, game-level, or fixed-length chunks of flattened packets?
2. Does the model see side-to-move as explicit `WHITE` / `BLACK` tokens or as a separate feature?
3. Does the output head predict all 8 slots jointly or sequentially?
4. Do we include game boundary tokens?
5. Do we include results as prediction targets or separate labels?
6. Do we need board-state-derived auxiliary features?
7. How will noising work for diffusion: token replacement, masking, categorical corruption, or embedding-space noise?

Until those are clearer, the full Lumbras tokenization should wait.

## Current recommended next step

Before running full dataset tokenization, decide the network contract:

```text
input representation
context window size
output representation
loss masking
legal move integration strategy
```

Once that is settled, we can tokenize the full base and fine-tune PGNs into the exact training format rather than producing a large intermediate that may need to be replaced.
