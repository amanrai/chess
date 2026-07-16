#!/usr/bin/env python3
"""Report verifier game lengths in fixed full-move buckets.

This reads ``offsets.npy`` only, so it can run against the existing verifier
store without loading move packets. By default, each bucket spans five full
moves (ten plies), which is the starting point for the planned 20M board-state
sample allocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "lumbras" / "verifier"


def length_summary(lengths: np.ndarray) -> dict[str, float | int]:
    if len(lengths) == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p10": float(np.percentile(lengths, 10)),
        "p90": float(np.percentile(lengths, 90)),
    }


def build_histogram(lengths: np.ndarray, bucket_moves: int) -> list[dict[str, int]]:
    """Return contiguous game-length bins, each spanning ``bucket_moves`` moves.

    The first bin includes zero through ``2 * bucket_moves`` plies. Subsequent
    bins start one ply after the preceding bin, so a game belongs to exactly one
    bin even when it ends after an odd number of plies.
    """
    if bucket_moves < 1:
        raise ValueError("bucket_moves must be >= 1")
    if len(lengths) == 0:
        return []

    width_plies = bucket_moves * 2
    # Zero-length games share bucket zero; positive lengths 1..width also map there.
    bucket_indices = np.maximum(lengths - 1, 0) // width_plies
    counts = np.bincount(bucket_indices.astype(np.int64))
    records = []
    for index, count in enumerate(counts):
        if not count:
            continue
        lo_plies = 0 if index == 0 else index * width_plies + 1
        hi_plies = (index + 1) * width_plies
        records.append(
            {
                "bucket_index": int(index),
                "moves_lo": int(np.ceil(lo_plies / 2)),
                "moves_hi": int(hi_plies // 2),
                "plies_lo": int(lo_plies),
                "plies_hi": int(hi_plies),
                "games": int(count),
            }
        )
    return records


def print_histogram(records: list[dict[str, int]], total_games: int) -> None:
    print(f"{'moves':>11}  {'plies':>11}  {'games':>12}  {'percent':>8}  {'cumulative':>10}")
    cumulative = 0
    for record in records:
        cumulative += record["games"]
        percent = 100 * record["games"] / total_games if total_games else 0.0
        cumulative_percent = 100 * cumulative / total_games if total_games else 0.0
        print(
            f"{record['moves_lo']:>4}-{record['moves_hi']:<4}  "
            f"{record['plies_lo']:>4}-{record['plies_hi']:<4}  "
            f"{record['games']:>12,}  {percent:>7.3f}%  {cumulative_percent:>9.3f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--bucket-moves",
        type=int,
        default=5,
        help="Full moves per histogram bucket; each move is two plies (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for later sampling-strategy work",
    )
    args = parser.parse_args()

    if args.bucket_moves < 1:
        raise ValueError("--bucket-moves must be >= 1")
    offsets_path = args.data_dir / "offsets.npy"
    if not offsets_path.exists():
        raise FileNotFoundError(f"missing verifier offsets: {offsets_path}")

    offsets = np.load(offsets_path, mmap_mode="r")
    if len(offsets) < 1:
        raise ValueError(f"offsets must contain at least one entry: {offsets_path}")
    lengths = np.diff(offsets)
    records = build_histogram(lengths, args.bucket_moves)
    summary = length_summary(lengths)
    print(
        "game lengths: "
        f"games={len(lengths):,} bucket_moves={args.bucket_moves} "
        f"bucket_plies={args.bucket_moves * 2}"
    )
    print(
        "summary: "
        f"min={summary['min']} max={summary['max']} mean={summary['mean']:.2f} "
        f"median={summary['median']:.2f} p10={summary['p10']:.2f} p90={summary['p90']:.2f}"
    )
    print_histogram(records, len(lengths))

    if args.output is not None:
        payload = {
            "data_dir": str(args.data_dir),
            "offsets_path": str(offsets_path),
            "games": int(len(lengths)),
            "bucket_moves": args.bucket_moves,
            "bucket_plies": args.bucket_moves * 2,
            "summary_plies": summary,
            "buckets": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote histogram JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
