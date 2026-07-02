#!/usr/bin/env python3
"""Collect every distinct character used in PGN movetext, excluding metadata.

PGN files contain tag-pair metadata lines like:
  [Event "..."]

This script ignores those metadata lines and only scans the game/movetext
section: move numbers, SAN moves, comments, NAGs, variations, results, and any
other non-header movetext characters present in the file.

By default it writes a JSON vocabulary summary to:
  data/processed/lumbras/movetext_chars.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUTS = [
    Path("data/processed/lumbras/lumbras_otb_both_2200_to_2399_base.pgn"),
    Path("data/processed/lumbras/lumbras_otb_both_2400_plus_ft.pgn"),
]
DEFAULT_OUTPUT = Path("data/processed/lumbras/movetext_chars.json")


def is_metadata_line(line: str) -> bool:
    # PGN tag-pair metadata starts in column 1 as [Key "Value"]. Variations in
    # movetext can also use brackets rarely in comments, but those are not tag
    # pair lines because they do not start with [AlphaKey + space + quote.
    if not line.startswith("["):
        return False
    close = line.find(" ")
    return close > 1 and line[1:close].replace("_", "").isalnum() and line[close + 1 : close + 2] == '"'


def display_char(ch: str) -> str:
    names = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        " ": "space",
    }
    return names.get(ch, ch)


def collect(paths: list[Path], limit_lines: int = 0) -> dict:
    counts: Counter[str] = Counter()
    total_movetext_chars = 0
    metadata_lines = 0
    movetext_lines = 0
    blank_lines = 0
    files: dict[str, dict[str, int]] = {}

    for path in paths:
        file_counts: Counter[str] = Counter()
        file_movetext_chars = 0
        file_metadata_lines = 0
        file_movetext_lines = 0
        file_blank_lines = 0
        lines_seen = 0

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lines_seen += 1
                if limit_lines and lines_seen > limit_lines:
                    break

                if is_metadata_line(line):
                    metadata_lines += 1
                    file_metadata_lines += 1
                    continue

                if line.strip() == "":
                    blank_lines += 1
                    file_blank_lines += 1
                    continue

                movetext_lines += 1
                file_movetext_lines += 1
                counts.update(line)
                file_counts.update(line)
                total_movetext_chars += len(line)
                file_movetext_chars += len(line)

        files[str(path)] = {
            "metadata_lines": file_metadata_lines,
            "movetext_lines": file_movetext_lines,
            "blank_lines": file_blank_lines,
            "movetext_chars": file_movetext_chars,
            "distinct_chars": len(file_counts),
        }

    chars = sorted(counts)
    return {
        "inputs": [str(path) for path in paths],
        "metadata_lines_ignored": metadata_lines,
        "movetext_lines_scanned": movetext_lines,
        "blank_lines_ignored": blank_lines,
        "movetext_chars_scanned": total_movetext_chars,
        "distinct_char_count": len(chars),
        "chars": chars,
        "chars_display": [display_char(ch) for ch in chars],
        "char_counts": {ch: counts[ch] for ch in chars},
        "char_counts_display": {display_char(ch): counts[ch] for ch in chars},
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-lines", type=int, default=0, help="Optional per-file line limit for smoke tests")
    parser.add_argument("--print-chars", action="store_true", help="Print the character list to stdout")
    args = parser.parse_args()

    missing = [path for path in args.inputs if not path.exists()]
    if missing:
        raise SystemExit("Missing input file(s): " + ", ".join(str(path) for path in missing))

    result = collect(args.inputs, args.limit_lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Scanned {result['movetext_chars_scanned']:,} movetext characters")
    print(f"Found {result['distinct_char_count']:,} distinct movetext characters")
    print(f"Wrote {args.output}")
    if args.print_chars:
        print("".join(result["chars"]))
        print(result["chars_display"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
