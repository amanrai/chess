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
            if args.max_games and len(pgn_texts) >= args.max_games:
                break
        if args.max_games and len(pgn_texts) >= args.max_games:
            break

    if not moves_chunks:
        raise SystemExit("No eligible games found")
    print(f"accepted {len(pgn_texts):,} games; skipped={dict(skipped)}")
    return np.concatenate(moves_chunks, axis=0), np.asarray(offsets, dtype=np.int64), pgn_texts, headers_list


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


def build_samples(offsets: np.ndarray, nonterminal: int, terminal: int, bucket_plies: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    nonterminal_buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    terminal_buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for game_id in range(len(offsets) - 1):
        length = int(offsets[game_id + 1] - offsets[game_id])
        for ply in range(1, length):
            nonterminal_buckets[(ply - 1) // bucket_plies].append((game_id, ply))
        terminal_buckets[(length - 1) // bucket_plies].append((game_id, length))

    nonterm = sample_equal_from_buckets(nonterminal_buckets, nonterminal, rng)
    term = sample_equal_from_buckets(terminal_buckets, terminal, rng)
    rows = np.empty(len(nonterm) + len(term), dtype=METADATA_DTYPE)
    for i, (game_id, ply) in enumerate(nonterm):
        rows[i] = (game_id, ply, False)
    for j, (game_id, ply) in enumerate(term, start=len(nonterm)):
        rows[j] = (game_id, ply, True)
    rng.shuffle(rows)
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-games", type=int, default=0, help="Debug cap on eligible games; 0 keeps all")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    games_dir = args.out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"BSE checkpoint: {checkpoint}")
    model, ckpt = load_encoder(checkpoint, args.device)
    ckpt_args = ckpt.get("args", {})
    context_plies = int(ckpt_args.get("context_plies", 128))
    pad_id = int(ckpt_args.get("pad_id", 0))

    moves, offsets, pgn_texts, headers = read_eligible_games(args)
    np.save(games_dir / "moves.npy", moves)
    np.save(games_dir / "offsets.npy", offsets)
    with (games_dir / "pgn_texts.jsonl").open("w", encoding="utf-8") as handle:
        for game_id, pgn in enumerate(pgn_texts):
            handle.write(json.dumps({"game_id": game_id, "pgn": pgn}) + "\n")
    with (games_dir / "game_headers.jsonl").open("w", encoding="utf-8") as handle:
        for game_id, h in enumerate(headers):
            handle.write(json.dumps({"game_id": game_id, "headers": h}) + "\n")

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
