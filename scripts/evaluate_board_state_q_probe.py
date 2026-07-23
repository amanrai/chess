#!/usr/bin/env python3
"""Evaluate a board-state Q-probe checkpoint on all-64-square reconstruction.

Loads an eval store produced by ``build_board_state_eval_bucket.py`` and a
board-state Q-probe checkpoint. For every selected game and every stored ply,
the script encodes the move-history prefix, queries all 64 squares, compares to
packed python-chess board labels, and prints metrics in 5-ply buckets by
default.
"""
from __future__ import annotations

import argparse
import json
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

from chessgm.tokenizer import TOKEN_TO_ID, VOCAB  # noqa: E402
from train_board_state_q_probe import (  # noqa: E402
    EMPTY,
    NUM_OCCUPANTS,
    NUM_SQUARES,
    OCCUPANT_NAMES,
    QBoardStateProbeTransformer,
    format_table,
    unpack_board_labels,
)

DEFAULT_EVAL_DIR = ROOT / "data" / "processed" / "lumbras" / "eval_board_state_2000_2199"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "board_state_q_probe"


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort snapshots by epoch then batch, with end-of-epoch after snapshots."""
    name = path.name
    epoch_match = re.search(r"_epoch_(\d+)", name)
    batch_match = re.search(r"_batch_(\d+)", name)
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    batch = int(batch_match.group(1)) if batch_match else 10**12
    return epoch, batch, name


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
    if not candidates:
        raise SystemExit(f"No .pt checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        return path
    return latest_checkpoint(DEFAULT_CHECKPOINT_DIR)


def load_model(checkpoint_path: Path, device: str) -> tuple[QBoardStateProbeTransformer, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    required = ["model_dim", "heads", "history_layers", "q_layers", "num_queries", "dropout"]
    missing = [name for name in required if name not in args]
    if missing:
        raise ValueError(f"checkpoint args missing required model config fields: {missing}")
    model = QBoardStateProbeTransformer(
        vocab_size=len(VOCAB),
        ply_expr=int(args.get("ply_expr", 8)),
        model_dim=int(args["model_dim"]),
        heads=int(args["heads"]),
        history_layers=int(args["history_layers"]),
        q_layers=int(args["q_layers"]),
        num_queries=int(args["num_queries"]),
        dropout=float(args["dropout"]),
        pad_id=int(args.get("pad_id", 0)),
    )
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"], strict=False)
    if missing_keys or unexpected_keys:
        raise ValueError(f"checkpoint load mismatch: missing={missing_keys} unexpected={unexpected_keys}")
    model.to(device)
    model.eval()
    return model, ckpt


def strip_check_mate_tokens(prefix: np.ndarray, pad_id: int, ply_expr: int) -> np.ndarray:
    check_id = TOKEN_TO_ID["CHECK"]
    mate_id = TOKEN_TO_ID["MATE"]
    leak_label_ids = {check_id, mate_id}
    stripped = np.full(prefix.shape, pad_id, dtype=prefix.dtype)
    for row_i, row in enumerate(prefix):
        kept = [int(token_id) for token_id in row if int(token_id) not in leak_label_ids]
        kept = kept[:ply_expr]
        stripped[row_i, : len(kept)] = kept
    return stripped


def make_prefix_x(moves: np.ndarray, start: int, prefix_plies: int, context_plies: int, pad_id: int) -> np.ndarray:
    prefix = moves[start : start + prefix_plies]
    prefix_x = strip_check_mate_tokens(prefix, pad_id=pad_id, ply_expr=int(moves.shape[1]))
    if len(prefix_x) >= context_plies:
        return prefix_x[-context_plies:].astype(np.int64, copy=False)
    pad_rows = np.full((context_plies - len(prefix_x), moves.shape[1]), pad_id, dtype=np.uint16)
    return np.concatenate([pad_rows, prefix_x], axis=0).astype(np.int64, copy=False)


def selected_game_indices(num_games: int, max_games: int, seed: int) -> list[int]:
    indices = list(range(num_games))
    if max_games and max_games < num_games:
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_games))
    return indices


def iter_positions(offsets: np.ndarray, game_indices: Iterable[int]) -> Iterable[tuple[int, int, int]]:
    """Yield (game_i, prefix_plies, board_row)."""
    for game_i in game_indices:
        start = int(offsets[game_i])
        end = int(offsets[game_i + 1])
        for prefix_plies in range(1, end - start + 1):
            yield game_i, prefix_plies, start + prefix_plies - 1


def empty_stats() -> dict:
    return {
        "positions": 0,
        "squares": 0,
        "square_correct": 0,
        "exact_boards": 0,
        "occupied": 0,
        "pred_occupied": 0,
        "occupied_correct": 0,
        "wrong_square_hist": [0] * (NUM_SQUARES + 1),
    }


def update_stats(stats: dict, pred: np.ndarray, labels: np.ndarray) -> None:
    correct = pred == labels
    occupied = labels != EMPTY
    pred_occupied = pred != EMPTY
    wrong_counts = (~correct).sum(axis=1)
    stats["positions"] += int(labels.shape[0])
    stats["squares"] += int(labels.size)
    stats["square_correct"] += int(correct.sum())
    stats["exact_boards"] += int(correct.all(axis=1).sum())
    stats["occupied"] += int(occupied.sum())
    stats["pred_occupied"] += int(pred_occupied.sum())
    stats["occupied_correct"] += int((correct & occupied).sum())
    wrong_hist = np.bincount(wrong_counts, minlength=NUM_SQUARES + 1)
    for i, count in enumerate(wrong_hist):
        stats["wrong_square_hist"][i] += int(count)


def percentile_from_hist(hist: list[int], percentile: float) -> int:
    total = sum(hist)
    if total <= 0:
        return 0
    threshold = percentile * total
    cumulative = 0
    for value, count in enumerate(hist):
        cumulative += count
        if cumulative >= threshold:
            return value
    return len(hist) - 1


def metric_row(name: str, stats: dict) -> list[str]:
    sq_acc = stats["square_correct"] / stats["squares"] if stats["squares"] else 0.0
    exact = stats["exact_boards"] / stats["positions"] if stats["positions"] else 0.0
    occ_p = stats["occupied_correct"] / stats["pred_occupied"] if stats["pred_occupied"] else 0.0
    occ_r = stats["occupied_correct"] / stats["occupied"] if stats["occupied"] else 0.0
    avg_wrong_squares = (stats["squares"] - stats["square_correct"]) / stats["positions"] if stats["positions"] else 0.0
    p50_wrong = percentile_from_hist(stats["wrong_square_hist"], 0.50)
    p75_wrong = percentile_from_hist(stats["wrong_square_hist"], 0.75)
    p90_wrong = percentile_from_hist(stats["wrong_square_hist"], 0.90)
    return [
        name,
        str(stats["positions"]),
        f"{sq_acc:.4f}",
        f"{exact:.4f}",
        f"{avg_wrong_squares:.3f}",
        str(p50_wrong),
        str(p75_wrong),
        str(p90_wrong),
        f"{occ_p:.4f}",
        f"{occ_r:.4f}",
        str(stats["occupied"]),
        str(stats["pred_occupied"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to evaluate. Default: latest .pt in checkpoints/board_state_q_probe")
    parser.add_argument("--max-games", type=int, default=0, help="Maximum eval games to sample; 0 evaluates all games")
    parser.add_argument("--seed", type=int, default=20260723, help="Random seed used when --max-games samples a subset")
    parser.add_argument("--bucket-plies", type=int, default=5, help="Ply bucket width for metrics")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of positions/prefixes per forward pass")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write metrics JSON")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.max_games < 0:
        raise SystemExit("--max-games must be >= 0")
    if args.bucket_plies < 1:
        raise SystemExit("--bucket-plies must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model, ckpt = load_model(checkpoint_path, args.device)
    ckpt_args = ckpt.get("args", {})
    context_plies = int(ckpt_args.get("context_plies", 128))
    pad_id = int(ckpt_args.get("pad_id", 0))

    moves = np.load(args.eval_dir / "moves.npy", mmap_mode="r")
    offsets = np.load(args.eval_dir / "offsets.npy", mmap_mode="r")
    boards = np.load(args.eval_dir / "board_after_packed.npy", mmap_mode="r")
    manifest_path = args.eval_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if len(offsets) < 2:
        raise SystemExit("Eval offsets contain no games")
    if int(offsets[-1]) != len(moves) or len(boards) != len(moves):
        raise SystemExit("Eval arrays are inconsistent")

    game_indices = selected_game_indices(len(offsets) - 1, args.max_games, args.seed)
    total_positions = sum(int(offsets[i + 1] - offsets[i]) for i in game_indices)
    square_ids = torch.arange(NUM_SQUARES, dtype=torch.long, device=args.device)[None, :]

    overall = empty_stats()
    bucket_stats: dict[int, dict[str, int]] = defaultdict(empty_stats)
    class_correct = np.zeros(NUM_OCCUPANTS, dtype=np.int64)
    class_total = np.zeros(NUM_OCCUPANTS, dtype=np.int64)
    class_pred_total = np.zeros(NUM_OCCUPANTS, dtype=np.int64)

    pending_x: list[np.ndarray] = []
    pending_labels: list[np.ndarray] = []
    pending_buckets: list[int] = []

    def flush_batch() -> None:
        if not pending_x:
            return
        x = torch.from_numpy(np.stack(pending_x)).to(args.device, non_blocking=True)
        batch_square_ids = square_ids.expand(x.shape[0], -1)
        with torch.inference_mode():
            logits, _, _, _ = model(x, batch_square_ids)
            pred = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
        labels = np.stack(pending_labels).astype(np.int64, copy=False)
        update_stats(overall, pred, labels)
        for bucket in sorted(set(pending_buckets)):
            mask = np.asarray([b == bucket for b in pending_buckets], dtype=bool)
            update_stats(bucket_stats[bucket], pred[mask], labels[mask])
        class_total[:] += np.bincount(labels.reshape(-1), minlength=NUM_OCCUPANTS)
        class_pred_total[:] += np.bincount(pred.reshape(-1), minlength=NUM_OCCUPANTS)
        class_correct[:] += np.bincount(labels.reshape(-1)[(pred == labels).reshape(-1)], minlength=NUM_OCCUPANTS)
        pending_x.clear()
        pending_labels.clear()
        pending_buckets.clear()

    iterator = tqdm(
        iter_positions(offsets, game_indices),
        total=total_positions,
        desc="evaluating all-square board reconstruction",
        unit="position",
        disable=args.no_progress,
    )
    for _game_i, prefix_plies, board_row in iterator:
        start = int(offsets[_game_i])
        pending_x.append(make_prefix_x(moves, start, prefix_plies, context_plies, pad_id))
        pending_labels.append(unpack_board_labels(boards[board_row]).astype(np.int64, copy=False))
        pending_buckets.append((prefix_plies - 1) // args.bucket_plies)
        if len(pending_x) >= args.batch_size:
            flush_batch()
    flush_batch()

    bucket_rows = []
    bucket_metrics_json = []
    for bucket in sorted(bucket_stats):
        lo = bucket * args.bucket_plies + 1
        hi = (bucket + 1) * args.bucket_plies
        row = metric_row(f"{lo}-{hi}", bucket_stats[bucket])
        bucket_rows.append(row)
        stats = bucket_stats[bucket]
        bucket_metrics_json.append({
            "plies_lo": lo,
            "plies_hi": hi,
            **stats,
            "exact_square_acc": stats["square_correct"] / stats["squares"] if stats["squares"] else 0.0,
            "exact_board_acc": stats["exact_boards"] / stats["positions"] if stats["positions"] else 0.0,
            "avg_wrong_squares": (stats["squares"] - stats["square_correct"]) / stats["positions"] if stats["positions"] else 0.0,
            "p50_wrong_squares": percentile_from_hist(stats["wrong_square_hist"], 0.50),
            "p75_wrong_squares": percentile_from_hist(stats["wrong_square_hist"], 0.75),
            "p90_wrong_squares": percentile_from_hist(stats["wrong_square_hist"], 0.90),
            "occupied_precision": stats["occupied_correct"] / stats["pred_occupied"] if stats["pred_occupied"] else 0.0,
            "occupied_recall": stats["occupied_correct"] / stats["occupied"] if stats["occupied"] else 0.0,
        })

    class_rows = []
    class_metrics_json = []
    for cls, name in enumerate(OCCUPANT_NAMES):
        total = int(class_total[cls])
        pred_total = int(class_pred_total[cls])
        correct = int(class_correct[cls])
        precision = correct / pred_total if pred_total else 0.0
        recall = correct / total if total else 0.0
        class_rows.append([name, f"{precision:.4f}", f"{recall:.4f}", str(correct), str(total), str(pred_total)])
        class_metrics_json.append({"class_id": cls, "class": name, "precision": precision, "recall": recall, "correct": correct, "n": total, "pred_n": pred_total})

    print(f"checkpoint: {checkpoint_path}")
    print(f"eval_dir: {args.eval_dir}")
    print(f"eval manifest kind: {manifest.get('kind')}")
    print(f"games evaluated: {len(game_indices)} / {len(offsets) - 1}")
    print("\noverall")
    metric_headers = ["bucket", "positions", "exact_square", "exact_board", "avg_wrong_sq", "p50_wrong", "p75_wrong", "p90_wrong", "occ_p", "occ_r", "occ_n", "pred_occ_n"]
    print(format_table(metric_headers, [metric_row("all", overall)]))
    print("\nply buckets")
    print(format_table(["plies", *metric_headers[1:]], bucket_rows))
    print("\noccupant classes")
    print(format_table(["class", "precision", "recall", "correct", "n", "pred_n"], class_rows))

    if args.json_out is not None:
        payload = {
            "checkpoint": str(checkpoint_path),
            "eval_dir": str(args.eval_dir),
            "eval_manifest_kind": manifest.get("kind"),
            "games_evaluated": len(game_indices),
            "total_games": int(len(offsets) - 1),
            "max_games": args.max_games,
            "seed": args.seed,
            "bucket_plies": args.bucket_plies,
            "overall": {
                **overall,
                "exact_square_acc": overall["square_correct"] / overall["squares"] if overall["squares"] else 0.0,
                "exact_board_acc": overall["exact_boards"] / overall["positions"] if overall["positions"] else 0.0,
                "avg_wrong_squares": (overall["squares"] - overall["square_correct"]) / overall["positions"] if overall["positions"] else 0.0,
                "p50_wrong_squares": percentile_from_hist(overall["wrong_square_hist"], 0.50),
                "p75_wrong_squares": percentile_from_hist(overall["wrong_square_hist"], 0.75),
                "p90_wrong_squares": percentile_from_hist(overall["wrong_square_hist"], 0.90),
                "occupied_precision": overall["occupied_correct"] / overall["pred_occupied"] if overall["pred_occupied"] else 0.0,
                "occupied_recall": overall["occupied_correct"] / overall["occupied"] if overall["occupied"] else 0.0,
            },
            "buckets": bucket_metrics_json,
            "classes": class_metrics_json,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote metrics JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
