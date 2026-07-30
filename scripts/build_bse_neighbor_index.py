#!/usr/bin/env python3
"""Build a small flattened-BSE nearest-neighbor index for latent inspection.

This is an exploratory analysis builder, not a production data pipeline. It:

1. reads PGN files, keeping games where both Elo headers are in range;
2. tokenizes accepted games into the same move-packet format as verifier data;
3. samples positions:
     - non-terminal positions roughly equally across 5-ply buckets;
     - terminal positions separately, roughly equally by terminal game length bucket;
4. encodes each sampled prefix through a frozen board-state Q-former encoder;
5. writes raw and L2-normalized flattened fp16 banks for brute-force search.

Output layout:

  data/analysis/bse_neighbors_1800_2200_100k/
    config.json
    metadata.npy              structured: game_id, ply, is_terminal
    vectors_raw.fp16.npy      float16 [N, K * D]
    vectors_norm.fp16.npy     float16 [N, K * D]
    games/
      moves.npy               uint16 [total_plies, ply_expr]
      offsets.npy             int64 [num_games + 1]
      pgn_texts.jsonl         {game_id, pgn}
      game_headers.jsonl      {game_id, headers}
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from tqdm.auto import tqdm

from chessgm.tokenizer import ChessTokenizer, VOCAB
from evaluate_board_state_q_probe import checkpoint_sort_key
from preprocess_verifier_dataset import pgn_game_to_packets
from train_board_state_q_probe import QBoardStateProbeTransformer

DEFAULT_OUT_DIR = ROOT / "data" / "analysis" / "bse_neighbors_1800_2200_100k"
METADATA_DTYPE = np.dtype([("game_id", "<i8"), ("ply", "<i4"), ("is_terminal", "?")])
ELIGIBLE_GAME_FILES = ("moves.npy", "offsets.npy", "pgn_texts.jsonl", "game_headers.jsonl", "manifest.json")


def iter_pgn_games(path: Path) -> Iterable[str]:
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("[Event ") and buf:
                yield "".join(buf)
                buf = []
            buf.append(line)
        if buf:
            yield "".join(buf)


def parse_headers(game_text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in game_text.splitlines():
        if not line.startswith("["):
            break
        match = re.match(r'^\[([^\s]+) "(.*)"\]$', line)
        if match:
            headers[match.group(1)] = match.group(2)
    return headers


def int_header(headers: dict[str, str], key: str) -> int | None:
    value = headers.get(key)
    if value is None:
        return None
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def accepted_by_elo(headers: dict[str, str], min_elo: int, max_elo: int) -> bool:
    white = int_header(headers, "WhiteElo")
    black = int_header(headers, "BlackElo")
    return white is not None and black is not None and min_elo <= white <= max_elo and min_elo <= black <= max_elo


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
    if not candidates:
        raise SystemExit(f"No .pt checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def resolve_checkpoint(path: Path) -> Path:
    if path.is_dir():
        return latest_checkpoint(path)
    return path


def load_encoder(checkpoint_path: Path, device: str) -> tuple[QBoardStateProbeTransformer, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = QBoardStateProbeTransformer(
        vocab_size=len(VOCAB),
        ply_expr=int(args.get("ply_expr", 8)),
        model_dim=int(args["model_dim"]),
        heads=int(args["heads"]),
        history_layers=int(args["history_layers"]),
        q_layers=int(args["q_layers"]),
        num_queries=int(args["num_queries"]),
        dropout=float(args.get("dropout", 0.0)),
        pad_id=int(args.get("pad_id", 0)),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint load mismatch: missing={missing} unexpected={unexpected}")
    model.to(device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def strip_check_mate_tokens(prefix: np.ndarray, pad_id: int, ply_expr: int) -> np.ndarray:
    from chessgm.tokenizer import TOKEN_TO_ID

    leak = {TOKEN_TO_ID["CHECK"], TOKEN_TO_ID["MATE"]}
    stripped = np.full(prefix.shape, pad_id, dtype=prefix.dtype)
    for row_i, row in enumerate(prefix):
        kept = [int(tok) for tok in row if int(tok) not in leak]
        kept = kept[:ply_expr]
        stripped[row_i, : len(kept)] = kept
    return stripped


def make_prefix_x(moves: np.ndarray, start: int, prefix_plies: int, context_plies: int, pad_id: int) -> np.ndarray:
    prefix = moves[start : start + prefix_plies]
    prefix = strip_check_mate_tokens(prefix, pad_id=pad_id, ply_expr=int(moves.shape[1]))
    if len(prefix) >= context_plies:
        return prefix[-context_plies:].astype(np.int64, copy=False)
    pad = np.full((context_plies - len(prefix), moves.shape[1]), pad_id, dtype=np.uint16)
    return np.concatenate([pad, prefix], axis=0).astype(np.int64, copy=False)


def read_eligible_games(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    tokenizer = ChessTokenizer()
    moves_chunks: list[np.ndarray] = []
    offsets = [0]
    pgn_texts: list[str] = []
    headers_list: list[dict[str, str]] = []
    skipped = defaultdict(int)

    for pgn_path in args.pgn:
        iterator = tqdm(iter_pgn_games(pgn_path), desc=f"reading {pgn_path.name}", unit="game")
        for game_text in iterator:
            headers = parse_headers(game_text)
            if not accepted_by_elo(headers, args.min_elo, args.max_elo):
                skipped["elo"] += 1
                continue
            packets = pgn_game_to_packets(game_text, tokenizer, args.seq_len)
            if packets is None or len(packets) < 2:
                skipped["bad_or_short"] += 1
                continue
            game_id = len(pgn_texts)
            moves_chunks.append(packets)
            offsets.append(offsets[-1] + len(packets))
            pgn_texts.append(game_text)
            headers_list.append(headers)
            iterator.set_postfix(accepted=len(pgn_texts), skipped_elo=skipped["elo"])
            if args.max_games and len(pgn_texts) >= args.max_games:
                print(f"accepted game cap reached: {len(pgn_texts):,}/{args.max_games:,}")
                break
        if args.max_games and len(pgn_texts) >= args.max_games:
            break

    if not moves_chunks:
        raise SystemExit("No eligible games found")
    print(f"accepted {len(pgn_texts):,} games; skipped={dict(skipped)}")
    return np.concatenate(moves_chunks, axis=0), np.asarray(offsets, dtype=np.int64), pgn_texts, headers_list


def eligible_games_cache_exists(path: Path) -> bool:
    return all((path / name).is_file() for name in ELIGIBLE_GAME_FILES)


def load_eligible_games_cache(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    print(f"loading eligible games cache from {path}")
    moves = np.load(path / "moves.npy", mmap_mode="r")
    offsets = np.load(path / "offsets.npy", mmap_mode="r")
    pgn_texts: list[str] = []
    with (path / "pgn_texts.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pgn_texts.append(row["pgn"])
    headers: list[dict[str, str]] = []
    with (path / "game_headers.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            headers.append(row["headers"])
    print(f"loaded {len(pgn_texts):,} eligible games; total plies={len(moves):,}")
    return moves, offsets, pgn_texts, headers


def write_eligible_games_cache(
    path: Path,
    moves: np.ndarray,
    offsets: np.ndarray,
    pgn_texts: list[str],
    headers: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    print(f"writing eligible games cache to {path}")
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "moves.npy", moves)
    np.save(path / "offsets.npy", offsets)
    with (path / "pgn_texts.jsonl").open("w", encoding="utf-8") as handle:
        for game_id, pgn in enumerate(pgn_texts):
            handle.write(json.dumps({"game_id": game_id, "pgn": pgn}) + "\n")
    with (path / "game_headers.jsonl").open("w", encoding="utf-8") as handle:
        for game_id, h in enumerate(headers):
            handle.write(json.dumps({"game_id": game_id, "headers": h}) + "\n")
    manifest = {
        "kind": "bse_neighbor_eligible_games_cache_v1",
        "pgn": [str(path) for path in args.pgn],
        "min_elo": args.min_elo,
        "max_elo": args.max_elo,
        "seq_len": args.seq_len,
        "max_games": args.max_games,
        "num_games": len(pgn_texts),
        "total_plies": int(len(moves)),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def ensure_games_link(games_dir: Path, target: Path) -> None:
    if games_dir.exists() or games_dir.is_symlink():
        return
    games_dir.symlink_to(target.resolve(), target_is_directory=True)


def sample_equal_from_buckets(
    buckets: dict[int, list[tuple[int, int]]], total: int, rng: random.Random
) -> list[tuple[int, int]]:
    keys = sorted(k for k, v in buckets.items() if v)
    if not keys:
        return []
    base = total // len(keys)
    remainder = total % len(keys)
    selected: list[tuple[int, int]] = []
    leftovers: list[tuple[int, int]] = []
    for i, key in enumerate(keys):
        want = base + (1 if i < remainder else 0)
        items = buckets[key]
        if want <= 0:
            leftovers.extend(items)
        elif len(items) <= want:
            selected.extend(items)
        else:
            picked = rng.sample(items, want)
            picked_set = set(picked)
            selected.extend(picked)
            leftovers.extend(x for x in items if x not in picked_set)
    if len(selected) < total and leftovers:
        selected_set = set(selected)
        extra_pool = [x for x in leftovers if x not in selected_set]
        selected.extend(rng.sample(extra_pool, min(total - len(selected), len(extra_pool))))
    rng.shuffle(selected)
    return selected[:total]


def allocate_equal(capacities: dict[int, int], total: int) -> dict[int, int]:
    keys = sorted(k for k, cap in capacities.items() if cap > 0)
    if not keys or total <= 0:
        return {}
    want: dict[int, int] = {}
    base = total // len(keys)
    remainder = total % len(keys)
    for i, key in enumerate(keys):
        want[key] = min(capacities[key], base + (1 if i < remainder else 0))

    remaining = total - sum(want.values())
    while remaining > 0:
        open_keys = [key for key in keys if want[key] < capacities[key]]
        if not open_keys:
            break
        per_key = max(1, math.ceil(remaining / len(open_keys)))
        for key in open_keys:
            add = min(per_key, capacities[key] - want[key], remaining)
            want[key] += add
            remaining -= add
            if remaining <= 0:
                break
    return want


def build_samples(offsets: np.ndarray, nonterminal: int, terminal: int, bucket_plies: int, seed: int) -> np.ndarray:
    """Sample positions without materializing every legal (game, ply) pair.

    The full 1800-2200 cache has hundreds of millions of non-terminal plies, so
    the old list-of-tuples bucket construction could appear to hang or exhaust
    memory before encoding began. This version samples by bucket from vectorized
    per-game counts and only materializes the requested metadata rows.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    lengths = np.diff(offsets).astype(np.int64, copy=False)
    game_ids = np.arange(len(lengths), dtype=np.int64)

    nonterminal_capacities: dict[int, int] = {}
    max_len = int(lengths.max(initial=0))
    max_nonterminal_bucket = (max(0, max_len - 2) // bucket_plies) if max_len >= 2 else -1
    for bucket in range(max_nonterminal_bucket + 1):
        ply_lo = bucket * bucket_plies + 1
        ply_hi = (bucket + 1) * bucket_plies
        counts = np.maximum(np.minimum(lengths - 1, ply_hi) - ply_lo + 1, 0)
        total_count = int(counts.sum())
        if total_count > 0:
            nonterminal_capacities[bucket] = total_count

    terminal_bucket_ids = ((lengths - 1) // bucket_plies).astype(np.int64, copy=False)
    terminal_capacities = {
        int(bucket): int(count)
        for bucket, count in zip(*np.unique(terminal_bucket_ids, return_counts=True), strict=False)
        if int(count) > 0
    }

    nonterminal_wants = allocate_equal(nonterminal_capacities, nonterminal)
    terminal_wants = allocate_equal(terminal_capacities, terminal)
    total_rows = sum(nonterminal_wants.values()) + sum(terminal_wants.values())
    rows = np.empty(total_rows, dtype=METADATA_DTYPE)
    out_i = 0

    for bucket, want in sorted(nonterminal_wants.items()):
        if want <= 0:
            continue
        ply_lo = bucket * bucket_plies + 1
        ply_hi = (bucket + 1) * bucket_plies
        counts = np.maximum(np.minimum(lengths - 1, ply_hi) - ply_lo + 1, 0).astype(np.int64, copy=False)
        nonzero = counts > 0
        bucket_game_ids = game_ids[nonzero]
        bucket_counts = counts[nonzero]
        cumulative = np.cumsum(bucket_counts)
        total_count = int(cumulative[-1])
        ranks = np.asarray(rng.sample(range(total_count), min(want, total_count)), dtype=np.int64)
        ranks.sort()
        bucket_idx = np.searchsorted(cumulative, ranks, side="right")
        prev = np.zeros_like(ranks)
        mask = bucket_idx > 0
        prev[mask] = cumulative[bucket_idx[mask] - 1]
        sampled_game_ids = bucket_game_ids[bucket_idx]
        sampled_plies = ply_lo + (ranks - prev)
        n = len(sampled_game_ids)
        rows[out_i : out_i + n]["game_id"] = sampled_game_ids
        rows[out_i : out_i + n]["ply"] = sampled_plies.astype(np.int32, copy=False)
        rows[out_i : out_i + n]["is_terminal"] = False
        out_i += n

    for bucket, want in sorted(terminal_wants.items()):
        if want <= 0:
            continue
        candidates = game_ids[terminal_bucket_ids == bucket]
        picked = np_rng.choice(candidates, size=min(want, len(candidates)), replace=False).astype(np.int64, copy=False)
        n = len(picked)
        rows[out_i : out_i + n]["game_id"] = picked
        rows[out_i : out_i + n]["ply"] = lengths[picked].astype(np.int32, copy=False)
        rows[out_i : out_i + n]["is_terminal"] = True
        out_i += n

    rows = rows[:out_i]
    np_rng.shuffle(rows)
    print(
        f"sampled {len(rows):,} positions "
        f"({sum(nonterminal_wants.values()):,} nonterminal, {sum(terminal_wants.values()):,} terminal)"
    )
    return rows


def encode_vectors(
    model: QBoardStateProbeTransformer,
    moves: np.ndarray,
    offsets: np.ndarray,
    metadata: np.ndarray,
    out_dir: Path,
    *,
    batch_size: int,
    context_plies: int,
    pad_id: int,
    device: str,
) -> None:
    sample_x: list[np.ndarray] = []
    sample_indices: list[int] = []
    n = len(metadata)
    flat_dim = int(model.encoder.num_queries * model.encoder.model_dim)
    raw = np.lib.format.open_memmap(out_dir / "vectors_raw.fp16.npy", mode="w+", dtype=np.float16, shape=(n, flat_dim))
    norm = np.lib.format.open_memmap(out_dir / "vectors_norm.fp16.npy", mode="w+", dtype=np.float16, shape=(n, flat_dim))

    def flush() -> None:
        if not sample_x:
            return
        x = torch.from_numpy(np.stack(sample_x)).to(device, non_blocking=True)
        with torch.inference_mode():
            q = model.encoder(x).detach().float().reshape(x.shape[0], -1).cpu().numpy()
        q_norm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        raw[np.asarray(sample_indices)] = q.astype(np.float16)
        norm[np.asarray(sample_indices)] = q_norm.astype(np.float16)
        sample_x.clear()
        sample_indices.clear()

    for row_i, row in enumerate(tqdm(metadata, desc="encoding BSE prefixes", unit="pos")):
        game_id = int(row["game_id"])
        ply = int(row["ply"])
        start = int(offsets[game_id])
        sample_x.append(make_prefix_x(moves, start, ply, context_plies, pad_id))
        sample_indices.append(row_i)
        if len(sample_x) >= batch_size:
            flush()
    flush()
    raw.flush()
    norm.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, nargs="+", required=True, help="Input PGN files")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True, help="BSE checkpoint file or checkpoint directory")
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--max-elo", type=int, default=2200)
    parser.add_argument("--nonterminal-samples", type=int, default=90_000)
    parser.add_argument("--terminal-samples", type=int, default=10_000)
    parser.add_argument("--bucket-plies", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-games", type=int, default=0, help="Debug cap on eligible games; 0 keeps all")
    parser.add_argument(
        "--eligible-games-dir",
        type=Path,
        default=None,
        help="Reusable cache dir for tokenized eligible games; loaded when complete, otherwise created",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    games_dir = args.out_dir / "games"

    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"BSE checkpoint: {checkpoint}")
    model, ckpt = load_encoder(checkpoint, args.device)
    ckpt_args = ckpt.get("args", {})
    context_plies = int(ckpt_args.get("context_plies", 128))
    pad_id = int(ckpt_args.get("pad_id", 0))

    if args.eligible_games_dir is not None and eligible_games_cache_exists(args.eligible_games_dir):
        moves, offsets, pgn_texts, headers = load_eligible_games_cache(args.eligible_games_dir)
        ensure_games_link(games_dir, args.eligible_games_dir)
    else:
        moves, offsets, pgn_texts, headers = read_eligible_games(args)
        cache_dir = args.eligible_games_dir if args.eligible_games_dir is not None else games_dir
        write_eligible_games_cache(cache_dir, moves, offsets, pgn_texts, headers, args)
        if cache_dir != games_dir:
            ensure_games_link(games_dir, cache_dir)

    metadata = build_samples(offsets, args.nonterminal_samples, args.terminal_samples, args.bucket_plies, args.seed)
    np.save(args.out_dir / "metadata.npy", metadata)

    config = {
        "kind": "bse_flat_neighbor_index_v1",
        "checkpoint": str(checkpoint),
        "min_elo": args.min_elo,
        "max_elo": args.max_elo,
        "nonterminal_samples": args.nonterminal_samples,
        "terminal_samples": args.terminal_samples,
        "bucket_plies": args.bucket_plies,
        "eligible_games_dir": str(args.eligible_games_dir) if args.eligible_games_dir is not None else str(games_dir),
        "num_games": int(len(offsets) - 1),
        "total_plies": int(len(moves)),
        "num_positions": int(len(metadata)),
        "bank_shape": [int(model.encoder.num_queries), int(model.encoder.model_dim)],
        "flat_dim": int(model.encoder.num_queries * model.encoder.model_dim),
        "distance_formulae": {
            "l2": "d_i = ||x_i - q||_2^2 = sum_j (x_ij - q_j)^2; lower is nearer",
            "cosine": "xhat_i = x_i / max(||x_i||_2, eps), qhat = q / max(||q||_2, eps); score_i = dot(xhat_i, qhat); higher is nearer",
        },
        "metadata_fields": ["game_id", "ply", "is_terminal"],
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    encode_vectors(
        model,
        moves,
        offsets,
        metadata,
        args.out_dir,
        batch_size=args.batch_size,
        context_plies=context_plies,
        pad_id=pad_id,
        device=args.device,
    )
    print(f"wrote index to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
