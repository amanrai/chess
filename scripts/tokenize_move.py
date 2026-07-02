#!/usr/bin/env python3
"""Tokenize one SAN move into fixed-length token/id arrays."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessgm.tokenizer import ChessTokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("move", nargs="?", help="SAN move, e.g. Nbd7 or exd8=Q+")
    parser.add_argument("--seq-len", type=int, default=8, help="Fixed output length; default 8")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    move = args.move or input("SAN move> ").strip()
    tok = ChessTokenizer()
    encoded = tok.encode_move(move)

    if len(encoded.ids) > args.seq_len:
        raise SystemExit(f"Move needs {len(encoded.ids)} tokens, longer than seq-len={args.seq_len}: {encoded.tokens}")

    pad_id = tok.token_to_id["<PAD>"]
    tokens = encoded.tokens + ["<PAD>"] * (args.seq_len - len(encoded.tokens))
    ids = encoded.ids + [pad_id] * (args.seq_len - len(encoded.ids))
    decoded = tok.decode_move([i for i in ids if i != pad_id])

    payload = {
        "move": move,
        "seq_len": args.seq_len,
        "tokens": tokens,
        "ids": ids,
        "decoded": decoded,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"move:    {move}")
        print(f"tokens:  {tokens}")
        print(f"ids:     {ids}")
        print(f"decoded: {decoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
