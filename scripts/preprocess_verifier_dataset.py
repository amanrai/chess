#!/usr/bin/env python3
"""Preprocess PGNs into compact verifier training data.

Default output is a game-level token store:

  moves.npy    uint16[total_moves, seq_len]
  offsets.npy  int64[num_games + 1]
  results.npy  int64[num_games]  # 0 white win, 1 black win, 2 draw
  manifest.json

This format keeps game boundaries so verifier training can sample many different
prefixes from the same game without materializing every prefix.

Optional: --materialize-prefixes creates a fixed sampled prefix dataset too:

  prefix_x.npy uint16[num_examples, context_moves, seq_len]
  prefix_y.npy int64[num_examples]
  prefix_meta.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessgm.tokenizer import ChessTokenizer, VOCAB, split_token_moves  # noqa: E402

DEFAULT_INPUTS = [
    ROOT / "data" / "processed" / "lumbras" / "lumbras_otb_both_2200_to_2399_base.pgn",
    ROOT / "data" / "processed" / "lumbras" / "lumbras_otb_both_2400_plus_ft.pgn",
]
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "lumbras" / "verifier"

RESULT_TO_LABEL = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}
LABEL_TO_RESULT = {v: k for k, v in RESULT_TO_LABEL.items()}
RESULT_RE = re.compile(r"\s(1-0|0-1|1/2-1/2|\*)\s*$")
BucketMode = Literal["absolute", "fraction"]


@dataclass(frozen=True)
class PrefixBucket:
    name: str
    lo: float
    hi: float
    weight: float = 1.0


ABSOLUTE_PREFIX_BUCKETS = [
    PrefixBucket("opening_1_16", 1, 16, 1.0),
    PrefixBucket("early_mid_17_40", 17, 40, 1.0),
    PrefixBucket("midgame_41_80", 41, 80, 1.0),
    PrefixBucket("late_81_plus", 81, 10**9, 1.0),
]

FRACTION_PREFIX_BUCKETS = [
    PrefixBucket("first_20pct", 0.00, 0.20, 1.0),
    PrefixBucket("middle_30pct", 0.20, 0.50, 1.0),
    PrefixBucket("late_30pct", 0.50, 0.80, 1.0),
    PrefixBucket("final_20pct", 0.80, 1.00, 1.0),
]

_WORKER_TOKENIZER: ChessTokenizer | None = None


def _init_worker() -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = ChessTokenizer()


def _process_game_for_store(args: tuple[str, int]) -> tuple[int | None, np.ndarray | None, str | None]:
    game_text, seq_len = args
    tokenizer = _WORKER_TOKENIZER or ChessTokenizer()
    result = result_from_pgn_text(game_text)
    if result not in RESULT_TO_LABEL:
        return None, None, "unknown_result"
    packets = pgn_game_to_packets(game_text, tokenizer, seq_len)
    if packets is None or len(packets) < 2:
        return None, None, "empty_or_bad_game"
    return RESULT_TO_LABEL[result], packets, None


def count_pgn_games(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Event "):
                count += 1
    return count


def iter_pgn_games(path: Path):
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Event ") and buf:
                yield "".join(buf)
                buf = []
            buf.append(line)
        if buf:
            yield "".join(buf)


def result_from_pgn_text(game_text: str) -> str | None:
    for line in game_text.splitlines():
        match = re.match(r'^\[Result "(.*)"\]$', line)
        if match:
            return match.group(1)
    match = RESULT_RE.search(game_text.strip())
    return match.group(1) if match else None


def pgn_to_movetext(game_text: str) -> str:
    return "\n".join(line for line in game_text.splitlines() if not line.startswith("["))


def pgn_game_to_packets(game_text: str, tokenizer: ChessTokenizer, seq_len: int) -> np.ndarray | None:
    tokens = tokenizer.tokenize_movetext(pgn_to_movetext(game_text), include_turn_tokens=True)
    rows: list[list[int]] = []
    pad_id = tokenizer.token_to_id["<PAD>"]

    for move_tokens in split_token_moves(tokens):
        if move_tokens and move_tokens[0].startswith("RESULT_"):
            break
        ids = tokenizer.encode_tokens(move_tokens)
        if len(ids) > seq_len:
            raise ValueError(f"move packet exceeds seq_len={seq_len}: {move_tokens}")
        rows.append(ids + [pad_id] * (seq_len - len(ids)))

    if not rows:
        return None
    return np.asarray(rows, dtype=np.uint16)


def write_game_store(
    inputs: list[Path],
    out_dir: Path,
    seq_len: int,
    max_games: int = 0,
    progress: bool = True,
    workers: int = 1,
    chunksize: int = 64,
) -> dict:
    tokenizer = ChessTokenizer()
    all_moves: list[np.ndarray] = []
    offsets = [0]
    results: list[int] = []
    source_files: list[str] = []
    skipped = {"unknown_result": 0, "empty_or_bad_game": 0}

    def handle_processed(label: int | None, packets: np.ndarray | None, skipped_reason: str | None, path: Path) -> None:
        if skipped_reason:
            skipped[skipped_reason] += 1
            return
        assert label is not None
        assert packets is not None
        all_moves.append(packets)
        results.append(label)
        source_files.append(str(path))
        offsets.append(offsets[-1] + len(packets))

    for path in inputs:
        total_games = count_pgn_games(path) if progress else None
        if max_games:
            remaining = max(max_games - len(results), 0)
            total_games = min(total_games or remaining, remaining)
        desc = f"tokenizing {path.name}"

        if workers <= 1:
            game_iter = tqdm(iter_pgn_games(path), total=total_games, desc=desc, unit="game", disable=not progress)
            for game_text in game_iter:
                if max_games and len(results) >= max_games:
                    break
                label, packets, skipped_reason = _process_game_for_store((game_text, seq_len))
                handle_processed(label, packets, skipped_reason, path)
                game_iter.set_postfix(games=len(results), moves=offsets[-1], skipped=sum(skipped.values()))
        else:
            if progress:
                print(f"using {workers} worker processes, chunksize={chunksize}")
            game_args = ((game_text, seq_len) for game_text in iter_pgn_games(path))
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                mapped = pool.map(_process_game_for_store, game_args, chunksize=chunksize)
                game_iter = tqdm(mapped, total=total_games, desc=desc, unit="game", disable=not progress)
                for label, packets, skipped_reason in game_iter:
                    if max_games and len(results) >= max_games:
                        break
                    handle_processed(label, packets, skipped_reason, path)
                    game_iter.set_postfix(games=len(results), moves=offsets[-1], skipped=sum(skipped.values()))
        if max_games and len(results) >= max_games:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        print(f"concatenating {len(all_moves):,} games / {offsets[-1]:,} moves")
    moves = np.concatenate(all_moves, axis=0) if all_moves else np.empty((0, seq_len), dtype=np.uint16)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    results_arr = np.asarray(results, dtype=np.int64)
    game_lengths = np.diff(offsets_arr) if len(offsets_arr) > 1 else np.asarray([], dtype=np.int64)
    result_counts = {LABEL_TO_RESULT[i]: int((results_arr == i).sum()) for i in sorted(LABEL_TO_RESULT)}
    source_counts = {str(path): source_files.count(str(path)) for path in inputs}
    length_stats = {
        "min": int(game_lengths.min()) if len(game_lengths) else 0,
        "max": int(game_lengths.max()) if len(game_lengths) else 0,
        "mean": float(game_lengths.mean()) if len(game_lengths) else 0.0,
        "median": float(np.median(game_lengths)) if len(game_lengths) else 0.0,
        "p10": float(np.percentile(game_lengths, 10)) if len(game_lengths) else 0.0,
        "p90": float(np.percentile(game_lengths, 90)) if len(game_lengths) else 0.0,
    }

    if progress:
        print("dataset stats:")
        print(json.dumps({"result_counts": result_counts, "game_length_moves": length_stats, "source_counts": source_counts}, indent=2))
        print(f"writing arrays to {out_dir}")
    np.save(out_dir / "moves.npy", moves)
    np.save(out_dir / "offsets.npy", offsets_arr)
    np.save(out_dir / "results.npy", results_arr)

    manifest = {
        "kind": "verifier_game_store",
        "inputs": [str(p) for p in inputs],
        "seq_len": seq_len,
        "moves_path": str(out_dir / "moves.npy"),
        "offsets_path": str(out_dir / "offsets.npy"),
        "results_path": str(out_dir / "results.npy"),
        "num_games": int(len(results_arr)),
        "total_moves": int(len(moves)),
        "moves_shape": list(moves.shape),
        "offsets_shape": list(offsets_arr.shape),
        "results_shape": list(results_arr.shape),
        "dtype": {"moves": "uint16", "offsets": "int64", "results": "int64"},
        "label_map": {str(k): v for k, v in LABEL_TO_RESULT.items()},
        "pad_token": "<PAD>",
        "pad_id": tokenizer.token_to_id["<PAD>"],
        "eom_token": "<EOM>",
        "eom_id": tokenizer.token_to_id["<EOM>"],
        "vocab_size": len(VOCAB),
        "vocab": VOCAB,
        "skipped": skipped,
        "workers": workers,
        "chunksize": chunksize,
        "stats": {
            "result_counts": result_counts,
            "source_counts": source_counts,
            "game_length_moves": length_stats,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "sources.json").write_text(json.dumps(source_files, indent=2) + "\n", encoding="utf-8")
    return manifest


def bucket_to_prefix_range(num_moves: int, bucket: PrefixBucket, mode: BucketMode) -> tuple[int, int] | None:
    if num_moves <= 0:
        return None
    if mode == "absolute":
        lo = int(bucket.lo)
        hi = int(min(bucket.hi, num_moves))
    elif mode == "fraction":
        lo = max(1, int(num_moves * bucket.lo) + 1)
        hi = max(lo, int(num_moves * bucket.hi))
        hi = min(hi, num_moves)
    else:
        raise ValueError(f"unknown bucket mode: {mode}")
    if lo > num_moves or hi < lo:
        return None
    return lo, hi


def weighted_choice(items, weights, rng: random.Random):
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for item, weight in zip(items, weights):
        upto += weight
        if upto >= r:
            return item
    return items[-1]


def sample_prefix_length(num_moves: int, rng: random.Random, buckets: list[PrefixBucket], mode: BucketMode) -> tuple[int, str]:
    valid = [b for b in buckets if bucket_to_prefix_range(num_moves, b, mode) is not None]
    if not valid:
        return rng.randint(1, num_moves), "fallback_any"
    bucket = weighted_choice(valid, [b.weight for b in valid], rng)
    lo, hi = bucket_to_prefix_range(num_moves, bucket, mode)  # type: ignore[misc]
    return rng.randint(lo, hi), bucket.name


def materialize_prefix_examples(
    out_dir: Path,
    context_moves: int,
    prefixes_per_game: int,
    mode: BucketMode,
    seed: int,
    progress: bool = True,
) -> dict:
    moves = np.load(out_dir / "moves.npy", mmap_mode="r")
    offsets = np.load(out_dir / "offsets.npy")
    results = np.load(out_dir / "results.npy")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    pad_id = int(manifest["pad_id"])
    seq_len = int(manifest["seq_len"])
    buckets = FRACTION_PREFIX_BUCKETS if mode == "fraction" else ABSOLUTE_PREFIX_BUCKETS
    rng = random.Random(seed)

    n = len(results) * prefixes_per_game
    x = np.lib.format.open_memmap(out_dir / "prefix_x.npy", mode="w+", dtype=np.uint16, shape=(n, context_moves, seq_len))
    y = np.lib.format.open_memmap(out_dir / "prefix_y.npy", mode="w+", dtype=np.int64, shape=(n,))
    bucket_counts: dict[str, int] = {}
    prefix_lengths: list[int] = []
    meta_path = out_dir / "prefix_meta.jsonl"

    row = 0
    with meta_path.open("w", encoding="utf-8") as meta:
        game_iter = tqdm(enumerate(results), total=len(results), desc="materializing prefixes", unit="game", disable=not progress)
        for game_i, label in game_iter:
            start = int(offsets[game_i])
            end = int(offsets[game_i + 1])
            num_moves = end - start
            for _ in range(prefixes_per_game):
                t, bucket_name = sample_prefix_length(num_moves, rng, buckets, mode)
                prefix_start = start
                prefix_end = start + t
                prefix = moves[prefix_start:prefix_end]
                if len(prefix) >= context_moves:
                    arr = prefix[-context_moves:]
                else:
                    pad_rows = np.full((context_moves - len(prefix), seq_len), pad_id, dtype=np.uint16)
                    arr = np.concatenate([pad_rows, prefix], axis=0)
                x[row] = arr
                y[row] = int(label)
                meta.write(json.dumps({"row": row, "game_i": game_i, "prefix_len": t, "bucket": bucket_name, "label": int(label)}) + "\n")
                bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1
                prefix_lengths.append(int(t))
                row += 1
            game_iter.set_postfix(rows=row)

    if progress:
        print(f"flushing prefix arrays: {n:,} examples")
    x.flush()
    y.flush()
    prefix_length_arr = np.asarray(prefix_lengths, dtype=np.int64)
    prefix_stats = {
        "bucket_counts": bucket_counts,
        "prefix_length_moves": {
            "min": int(prefix_length_arr.min()) if len(prefix_length_arr) else 0,
            "max": int(prefix_length_arr.max()) if len(prefix_length_arr) else 0,
            "mean": float(prefix_length_arr.mean()) if len(prefix_length_arr) else 0.0,
            "median": float(np.median(prefix_length_arr)) if len(prefix_length_arr) else 0.0,
            "p10": float(np.percentile(prefix_length_arr, 10)) if len(prefix_length_arr) else 0.0,
            "p90": float(np.percentile(prefix_length_arr, 90)) if len(prefix_length_arr) else 0.0,
        },
    }
    if progress:
        print("prefix stats:")
        print(json.dumps(prefix_stats, indent=2))

    prefix_manifest = {
        "kind": "materialized_verifier_prefixes",
        "prefix_x_path": str(out_dir / "prefix_x.npy"),
        "prefix_y_path": str(out_dir / "prefix_y.npy"),
        "prefix_meta_path": str(meta_path),
        "shape_x": [n, context_moves, seq_len],
        "shape_y": [n],
        "context_moves": context_moves,
        "prefixes_per_game": prefixes_per_game,
        "bucket_mode": mode,
        "seed": seed,
        "stats": prefix_stats,
    }
    (out_dir / "prefix_manifest.json").write_text(json.dumps(prefix_manifest, indent=2) + "\n", encoding="utf-8")
    return prefix_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--max-games", type=int, default=0, help="Debug limit; 0 means all games")
    parser.add_argument("--materialize-prefixes", action="store_true")
    parser.add_argument("--context-moves", type=int, default=128)
    parser.add_argument("--prefixes-per-game", type=int, default=1)
    parser.add_argument("--bucket-mode", choices=["fraction", "absolute"], default="fraction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Game-tokenization worker processes; use e.g. --workers $(nproc)")
    parser.add_argument("--chunksize", type=int, default=64, help="ProcessPool map chunksize for --workers > 1")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        raise SystemExit("Missing input(s): " + ", ".join(str(p) for p in missing))

    progress = not args.no_progress
    workers = max(1, args.workers)
    if workers > (os.cpu_count() or workers) and progress:
        print(f"warning: workers={workers} exceeds detected cpu_count={os.cpu_count()}")
    manifest = write_game_store(
        args.inputs,
        args.out_dir,
        args.seq_len,
        args.max_games,
        progress=progress,
        workers=workers,
        chunksize=args.chunksize,
    )
    print(json.dumps({k: manifest[k] for k in ["num_games", "total_moves", "moves_shape", "skipped"]}, indent=2))

    if args.materialize_prefixes:
        prefix_manifest = materialize_prefix_examples(
            args.out_dir,
            context_moves=args.context_moves,
            prefixes_per_game=args.prefixes_per_game,
            mode=args.bucket_mode,
            seed=args.seed,
            progress=progress,
        )
        print(json.dumps(prefix_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
