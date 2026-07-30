#!/usr/bin/env python3
"""Benchmark brute-force top-1 BSE neighbor search over a flattened z1 index.

For each query, this script randomly selects an indexed position and finds its
nearest neighbor by both cosine and RMS/L2 over the whole index. It is intended to
measure whether brute-force search over the 1M index is viable for z2 dataset
construction.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

DEFAULT_INDEX_DIR = Path("/700gpart/chess/data/analysis/bse_neighbors_1800_2200_1m")


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def load_config(index_dir: Path) -> dict[str, Any]:
    path = index_dir / "config.json"
    if not path.exists():
        raise SystemExit(f"missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def choose_query_rows(n: int, queries: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if queries <= 0 or queries >= n:
        rows = np.arange(n, dtype=np.int64)
        rng.shuffle(rows)
        return rows
    return rng.choice(n, size=queries, replace=False).astype(np.int64, copy=False)


def top1_for_query(
    *,
    query_row: int,
    raw: np.ndarray,
    norm: np.ndarray,
    metadata: np.ndarray,
    chunk_size: int,
    device: str,
    exclude_self: bool,
    exclude_same_game: bool,
) -> dict[str, Any]:
    q_raw_np = np.asarray(raw[query_row], dtype=np.float32)
    q_norm_np = np.asarray(norm[query_row], dtype=np.float32)
    q_raw = torch.from_numpy(q_raw_np).to(device)
    q_norm = torch.from_numpy(q_norm_np).to(device)
    q_raw_norm_sq = float(torch.dot(q_raw, q_raw).item())
    query_game_id = int(metadata[query_row]["game_id"])
    flat_dim = int(raw.shape[1])

    best_cos = -float("inf")
    best_cos_i = -1
    best_l2 = float("inf")
    best_l2_i = -1

    for start in range(0, raw.shape[0], chunk_size):
        end = min(start + chunk_size, raw.shape[0])
        raw_chunk = torch.from_numpy(np.asarray(raw[start:end], dtype=np.float32)).to(device, non_blocking=True)
        norm_chunk = torch.from_numpy(np.asarray(norm[start:end], dtype=np.float32)).to(device, non_blocking=True)

        cos_scores = norm_chunk @ q_norm
        raw_norm_sq = torch.sum(raw_chunk * raw_chunk, dim=1)
        l2_scores = raw_norm_sq + q_raw_norm_sq - 2.0 * (raw_chunk @ q_raw)
        l2_scores = torch.clamp(l2_scores, min=0.0)

        if exclude_self and start <= query_row < end:
            local = query_row - start
            cos_scores[local] = -float("inf")
            l2_scores[local] = float("inf")

        if exclude_same_game:
            game_ids = metadata[start:end]["game_id"]
            same = game_ids == query_game_id
            if np.any(same):
                same_t = torch.from_numpy(same).to(device)
                cos_scores[same_t] = -float("inf")
                l2_scores[same_t] = float("inf")

        cos_val, cos_idx = torch.max(cos_scores, dim=0)
        l2_val, l2_idx = torch.min(l2_scores, dim=0)
        cos_float = float(cos_val.item())
        l2_float = float(l2_val.item())
        if cos_float > best_cos:
            best_cos = cos_float
            best_cos_i = start + int(cos_idx.item())
        if l2_float < best_l2:
            best_l2 = l2_float
            best_l2_i = start + int(l2_idx.item())

    return {
        "query_row": int(query_row),
        "query_game_id": query_game_id,
        "cosine_row": int(best_cos_i),
        "cosine": float(best_cos),
        "rms_row": int(best_l2_i),
        "l2_sq": float(best_l2),
        "rms": math.sqrt(max(best_l2, 0.0) / flat_dim),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--queries", type=int, default=0, help="Number of random query rows; 0 means every row once")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exclude-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-same-game", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL path for per-query top-1 results")
    args = parser.parse_args()

    t0 = time.perf_counter()
    index_dir = args.index_dir
    print(f"index dir: {index_dir}", flush=True)
    config = load_config(index_dir)
    print(f"config kind: {config.get('kind')}", flush=True)
    print(f"bank shape: {config.get('bank_shape')} flat_dim={config.get('flat_dim')}", flush=True)

    metadata_path = index_dir / "metadata.npy"
    raw_path = index_dir / "vectors_raw.fp16.npy"
    norm_path = index_dir / "vectors_norm.fp16.npy"
    for path in (metadata_path, raw_path, norm_path):
        if not path.exists():
            raise SystemExit(f"missing required index file: {path}")
        print(f"file: {path} size={sizeof_fmt(path.stat().st_size)}", flush=True)

    print("loading metadata", flush=True)
    metadata = np.load(metadata_path, mmap_mode="r", allow_pickle=False)
    print("opening vector memmaps", flush=True)
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    norm = np.load(norm_path, mmap_mode="r", allow_pickle=False)
    if raw.shape != norm.shape:
        raise SystemExit(f"raw/norm shape mismatch: {raw.shape} vs {norm.shape}")
    if len(metadata) != raw.shape[0]:
        raise SystemExit(f"metadata/vector row mismatch: {len(metadata)} vs {raw.shape[0]}")

    print(f"rows: {raw.shape[0]:,}", flush=True)
    print(f"dims: {raw.shape[1]:,}", flush=True)
    print(f"raw dtype: {raw.dtype}; norm dtype: {norm.dtype}", flush=True)
    print(f"metadata dtype: {metadata.dtype}", flush=True)
    print(f"device: {args.device}", flush=True)
    print(f"chunk size: {args.chunk_size:,}", flush=True)
    print(f"exclude self: {args.exclude_self}; exclude same game: {args.exclude_same_game}", flush=True)

    if args.device.startswith("cuda"):
        print(f"cuda device: {torch.cuda.get_device_name()}", flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f"cuda mem free/total: {sizeof_fmt(free)} / {sizeof_fmt(total)}", flush=True)

    query_rows = choose_query_rows(raw.shape[0], args.queries, args.seed)
    print(f"query count: {len(query_rows):,}", flush=True)
    print(f"pre-step elapsed: {time.perf_counter() - t0:.2f}s", flush=True)

    output_handle = args.output.open("w", encoding="utf-8") if args.output else None
    timings: list[float] = []
    examples: list[dict[str, Any]] = []
    search_start = time.perf_counter()

    try:
        pbar = tqdm(query_rows, desc="nearest-neighbor queries", unit="query")
        for query_i, row in enumerate(pbar, start=1):
            q0 = time.perf_counter()
            result = top1_for_query(
                query_row=int(row),
                raw=raw,
                norm=norm,
                metadata=metadata,
                chunk_size=args.chunk_size,
                device=args.device,
                exclude_self=args.exclude_self,
                exclude_same_game=args.exclude_same_game,
            )
            dt = time.perf_counter() - q0
            timings.append(dt)
            if len(examples) < 5:
                examples.append(result)
            if output_handle is not None:
                output_handle.write(json.dumps(result) + "\n")
            if query_i % max(1, args.log_every) == 0:
                elapsed = time.perf_counter() - search_start
                qps = query_i / max(elapsed, 1e-12)
                p50 = float(np.percentile(timings, 50))
                p90 = float(np.percentile(timings, 90))
                pbar.set_postfix(qps=f"{qps:.3f}", sec_p50=f"{p50:.2f}", sec_p90=f"{p90:.2f}")
    finally:
        if output_handle is not None:
            output_handle.close()

    elapsed = time.perf_counter() - search_start
    total_elapsed = time.perf_counter() - t0
    qps = len(query_rows) / max(elapsed, 1e-12)
    print("\nBenchmark complete")
    print(f"  queries: {len(query_rows):,}")
    print(f"  search elapsed: {elapsed:.2f}s")
    print(f"  total elapsed: {total_elapsed:.2f}s")
    print(f"  queries/sec: {qps:.4f}")
    if timings:
        print(f"  sec/query p50: {np.percentile(timings, 50):.4f}")
        print(f"  sec/query p90: {np.percentile(timings, 90):.4f}")
        print(f"  sec/query p99: {np.percentile(timings, 99):.4f}")
    if len(query_rows) < raw.shape[0] and qps > 0:
        projected = raw.shape[0] / qps
        print(f"  projected time for {raw.shape[0]:,} queries: {projected / 3600:.2f} hours")
    print("  examples:")
    for ex in examples:
        print("   ", json.dumps(ex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
