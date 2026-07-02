#!/usr/bin/env python3
"""Convert PGN movetext into fixed-width move-token ID arrays.

Output is a .npy array shaped [num_moves, seq_len], dtype uint16.
Each row is one complete move packet padded with <PAD>, e.g.:
  Nbd7 -> [PIECE_N, SRC_FILE_b, TO_d7, <EOM>, <PAD>, ...]

This is a notation-token training table, not board-state/legal-move data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessgm.tokenizer import ChessTokenizer, split_token_moves, tokenize_movetext, VOCAB  # noqa: E402

DEFAULT_INPUTS = [
    Path("data/processed/lumbras/lumbras_otb_both_2200_to_2399_base.pgn"),
    Path("data/processed/lumbras/lumbras_otb_both_2400_plus_ft.pgn"),
]
DEFAULT_OUT = Path("data/processed/lumbras/tokenized_moves_u16_seq8.npy")


def iter_game_movetext(path: Path):
    lines: list[str] = []
    in_game = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Event "):
                if lines:
                    yield "".join(lines)
                    lines = []
                in_game = True
                continue
            if not in_game:
                continue
            if line.startswith("["):
                continue
            if line.strip():
                lines.append(line)
        if lines:
            yield "".join(lines)


def iter_move_rows(paths: list[Path], tokenizer: ChessTokenizer, seq_len: int):
    pad_id = tokenizer.token_to_id["<PAD>"]
    for path in paths:
        for movetext in iter_game_movetext(path):
            tokens = tokenize_movetext(movetext, include_turn_tokens=True)
            for move_tokens in split_token_moves(tokens):
                # Results are game labels, not move packets for this table.
                if move_tokens and move_tokens[0].startswith("RESULT_"):
                    continue
                ids = tokenizer.encode_tokens(move_tokens)
                if len(ids) > seq_len:
                    raise ValueError(f"Move packet exceeds seq_len={seq_len}: {move_tokens}")
                yield ids + [pad_id] * (seq_len - len(ids))


def count_rows(paths: list[Path], tokenizer: ChessTokenizer, seq_len: int, limit: int = 0) -> int:
    count = 0
    for _ in iter_move_rows(paths, tokenizer, seq_len):
        count += 1
        if limit and count >= limit:
            break
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--limit-moves", type=int, default=0, help="Smoke-test limit; 0 means all moves")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        raise SystemExit("Missing input(s): " + ", ".join(str(p) for p in missing))

    tokenizer = ChessTokenizer()
    n_rows = count_rows(args.inputs, tokenizer, args.seq_len, args.limit_moves)
    print(f"move rows: {n_rows:,}")
    if args.count_only:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.uint16, shape=(n_rows, args.seq_len))

    for i, row in enumerate(iter_move_rows(args.inputs, tokenizer, args.seq_len)):
        if args.limit_moves and i >= args.limit_moves:
            break
        arr[i] = row
        if i and i % 1_000_000 == 0:
            print(f"wrote {i:,} rows")
    arr.flush()

    manifest = {
        "inputs": [str(p) for p in args.inputs],
        "output": str(args.output),
        "shape": [n_rows, args.seq_len],
        "dtype": "uint16",
        "seq_len": args.seq_len,
        "pad_token": "<PAD>",
        "pad_id": tokenizer.token_to_id["<PAD>"],
        "eom_token": "<EOM>",
        "eom_id": tokenizer.token_to_id["<EOM>"],
        "vocab_size": len(VOCAB),
        "vocab": VOCAB,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
