#!/usr/bin/env python3
"""Extract Lumbras OTB 2200+ games into base and fine-tune PGN splits.

Splits:
  - base: both players have Elo in [2200, 2400)
  - ft:   both players have Elo >= 2400

The script streams directly from the downloaded .7z archives with bsdtar, so it
never needs to extract the full corpus to disk first.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable

HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]')
DEFAULT_RAW_DIR = Path("data/raw/lumbras/otb")
DEFAULT_OUT_DIR = Path("data/processed/lumbras")


def parse_elo(value: str | None) -> int | None:
    if not value or value in {"?", "-"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def archive_stream(path: Path) -> Iterable[str]:
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
    closed_early = False
    try:
        yield from proc.stdout
    except GeneratorExit:
        closed_early = True
        proc.terminate()
        raise
    finally:
        stderr = proc.stderr.read() if proc.stderr else ""
        code = proc.wait()
        if code != 0 and not closed_early:
            raise RuntimeError(f"bsdtar failed for {path}: {stderr}")


def iter_games(path: Path) -> Iterable[tuple[dict[str, str], str]]:
    headers: dict[str, str] = {}
    lines: list[str] = []

    def flush() -> tuple[dict[str, str], str] | None:
        if not lines:
            return None
        text = "".join(lines)
        if not text.endswith("\n"):
            text += "\n"
        return headers, text

    for line in archive_stream(path):
        if line.startswith("[Event ") and lines:
            game = flush()
            if game is not None:
                yield game
            headers = {}
            lines = []
        lines.append(line)
        if line.startswith("["):
            match = HEADER_RE.match(line.rstrip("\n"))
            if match:
                headers[match.group(1)] = match.group(2)

    game = flush()
    if game is not None:
        yield game


def classify(headers: dict[str, str], min_elo: int, ft_elo: int) -> str | None:
    white = parse_elo(headers.get("WhiteElo"))
    black = parse_elo(headers.get("BlackElo"))
    if white is None or black is None:
        return None
    floor = min(white, black)
    if floor < min_elo:
        return None
    if floor >= ft_elo:
        return "ft"
    return "base"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-elo", type=int, default=2200)
    parser.add_argument("--ft-elo", type=int, default=2400)
    parser.add_argument("--limit", type=int, default=0, help="Optional max games to scan, for smoke tests")
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()

    archives = args.archives or sorted(args.raw_dir.glob("*.7z"))
    if not archives:
        raise SystemExit(f"No .7z archives found in {args.raw_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.out_dir / f"lumbras_otb_both_{args.min_elo}_to_{args.ft_elo - 1}_base.pgn"
    ft_path = args.out_dir / f"lumbras_otb_both_{args.ft_elo}_plus_ft.pgn"
    manifest_path = args.out_dir / f"lumbras_otb_{args.min_elo}_splits_manifest.json"

    counts: Counter[str] = Counter()
    per_archive: dict[str, dict[str, int]] = {}

    with base_path.open("w", encoding="utf-8") as base_out, ft_path.open("w", encoding="utf-8") as ft_out:
        for archive in archives:
            local: Counter[str] = Counter()
            print(f"Scanning {archive}")
            for headers, game_text in iter_games(archive):
                counts["seen"] += 1
                local["seen"] += 1
                split = classify(headers, args.min_elo, args.ft_elo)
                if split is None:
                    counts["skipped"] += 1
                    local["skipped"] += 1
                elif split == "base":
                    base_out.write(game_text)
                    base_out.write("\n")
                    counts["base"] += 1
                    local["base"] += 1
                elif split == "ft":
                    ft_out.write(game_text)
                    ft_out.write("\n")
                    counts["ft"] += 1
                    local["ft"] += 1
                else:
                    raise AssertionError(split)

                if args.limit and counts["seen"] >= args.limit:
                    break
            per_archive[archive.name] = dict(local)
            print(f"  seen={local['seen']:,} base={local['base']:,} ft={local['ft']:,} skipped={local['skipped']:,}")
            if args.limit and counts["seen"] >= args.limit:
                break

    manifest = {
        "source": "Lumbra's Gigabase OTB PGN archives",
        "min_elo": args.min_elo,
        "ft_elo": args.ft_elo,
        "base_definition": f"both players have Elo >= {args.min_elo} and < {args.ft_elo}",
        "ft_definition": f"both players have Elo >= {args.ft_elo}",
        "base_path": str(base_path),
        "ft_path": str(ft_path),
        "counts": dict(counts),
        "per_archive": per_archive,
        "archives": [str(path) for path in archives],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nDone")
    print(f"  base: {counts['base']:,} games -> {base_path}")
    print(f"  ft:   {counts['ft']:,} games -> {ft_path}")
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
