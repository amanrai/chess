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
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

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


def write_game_store(inputs: list[Path], out_dir: Path, seq_len: int, max_games: int = 0) -> dict:
    tokenizer = ChessTokenizer()
    all_moves: list[np.ndarray] = []
    offsets = [0]
    results: list[int] = []
    source_files: list[str] = []
    skipped = {"unknown_result": 0, "empty_or_bad_game": 0}

    for path in inputs:
        for game_text in iter_pgn_games(path):
            if max_games and len(results) >= max_games:
                break
            result = result_from_pgn_text(game_text)
            if result not in RESULT_TO_LABEL:
                skipped["unknown_result"] += 1
                continue
            packets = pgn_game_to_packets(game_text, tokenizer, seq_len)
            if packets is None or len(packets) < 2:
                skipped["empty_or_bad_game"] += 1
                continue
            all_moves.append(packets)
            results.append(RESULT_TO_LABEL[result])
            source_files.append(str(path))
            offsets.append(offsets[-1] + len(packets))
        if max_games and len(results) >= max_games:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    moves = np.concatenate(all_moves, axis=0) if all_moves else np.empty((0, seq_len), dtype=np.uint16)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    results_arr = np.asarray(results, dtype=np.int64)

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


def materialize_prefix_examples(out_dir: Path, context_moves: int, prefixes_per_game: int, mode: BucketMode, seed: int) -> dict:
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
    meta_path = out_dir / "prefix_meta.jsonl"

    row = 0
    with meta_path.open("w", encoding="utf-8") as meta:
        for game_i, label in enumerate(results):
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
                row += 1

    x.flush()
    y.flush()
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
    args = parser.parse_args()

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        raise SystemExit("Missing input(s): " + ", ".join(str(p) for p in missing))

    manifest = write_game_store(args.inputs, args.out_dir, args.seq_len, args.max_games)
    print(json.dumps({k: manifest[k] for k in ["num_games", "total_moves", "moves_shape", "skipped"]}, indent=2))

    if args.materialize_prefixes:
        prefix_manifest = materialize_prefix_examples(
            args.out_dir,
            context_moves=args.context_moves,
            prefixes_per_game=args.prefixes_per_game,
            mode=args.bucket_mode,
            seed=args.seed,
        )
        print(json.dumps(prefix_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
