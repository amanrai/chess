#!/usr/bin/env python3
"""Copy the first N games from a large PGN into a small sample PGN."""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_INPUT = Path("data/processed/lumbras/lumbras_otb_both_2400_plus_ft.pgn")
DEFAULT_OUTPUT = Path("data/processed/lumbras/sample_games.pgn")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-n", "--games", type=int, default=20)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing input: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    games_written = 0
    seen_movetext = False
    started = False

    with args.input.open("r", encoding="utf-8", errors="replace") as src, args.output.open("w", encoding="utf-8") as out:
        for line in src:
            if line.startswith("[Event "):
                if started and seen_movetext:
                    games_written += 1
                    if games_written >= args.games:
                        break
                    out.write("\n")
                started = True
                seen_movetext = False
            if started:
                out.write(line)
                if line.strip() and not line.startswith("["):
                    seen_movetext = True
        else:
            if started and seen_movetext:
                games_written += 1

    print(f"Wrote {games_written} game(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
