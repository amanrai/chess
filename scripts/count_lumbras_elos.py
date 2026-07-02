#!/usr/bin/env python3
"""Stream Lumbras PGN archives and count Elo threshold buckets."""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]')
DEFAULT_THRESHOLDS = [1800, 1900, 2000, 2100, 2200, 2300, 2400]


def parse_int(value: str | None) -> int | None:
    if not value or value in {"?", "-"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def scan_archive(path: Path, thresholds: list[int]) -> Counter:
    counts: Counter = Counter()
    headers: dict[str, str] = {}

    proc = subprocess.Popen(
        ["bsdtar", "-xOf", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1024 * 1024,
    )
    assert proc.stdout is not None

    def finish_game() -> None:
        if not headers:
            return
        counts["games"] += 1
        white = parse_int(headers.get("WhiteElo"))
        black = parse_int(headers.get("BlackElo"))
        if white is not None:
            counts["white_elo"] += 1
        if black is not None:
            counts["black_elo"] += 1
        if white is not None and black is not None:
            counts["both_elo"] += 1
            floor = min(white, black)
            for threshold in thresholds:
                if floor >= threshold:
                    counts[f"both_ge_{threshold}"] += 1

    for line in proc.stdout:
        if line.startswith("[Event "):
            finish_game()
            headers = {}
        if line.startswith("["):
            match = HEADER_RE.match(line.rstrip("\n"))
            if match:
                headers[match.group(1)] = match.group(2)
    finish_game()
    stderr = proc.stderr.read() if proc.stderr else ""
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"bsdtar failed for {path}: {stderr}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="*", type=Path, default=sorted(Path("data/raw/lumbras/otb").glob("*.7z")))
    parser.add_argument("--threshold", dest="thresholds", type=int, action="append")
    args = parser.parse_args()
    thresholds = args.thresholds or DEFAULT_THRESHOLDS

    total: Counter = Counter()
    for archive in args.archives:
        counts = scan_archive(archive, thresholds)
        total.update(counts)
        print(archive.name)
        print("  games", counts["games"])
        print("  both_elo", counts["both_elo"])
        for threshold in thresholds:
            print(f"  both_ge_{threshold}", counts[f"both_ge_{threshold}"])

    print("TOTAL")
    print("  games", total["games"])
    print("  both_elo", total["both_elo"])
    for threshold in thresholds:
        print(f"  both_ge_{threshold}", total[f"both_ge_{threshold}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
