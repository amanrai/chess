#!/usr/bin/env python3
"""Convert existing BSE neighbor .npy vectors into FAISS indexes.

This does not run the BSE model. It consumes the already-generated fp16 vector
files from scripts/build_bse_neighbor_index.py and writes FAISS indexes for fast
nearest-neighbor lookup.

Defaults build compact approximate IVF-PQ indexes:

  - cosine/IP: vectors_norm.fp16.npy -> faiss_cosine_ivfpq.index
  - L2/RMS:    vectors_raw.fp16.npy  -> faiss_l2_ivfpq.index

The FAISS row ids are implicit and match the row numbers in metadata.npy.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

DEFAULT_INDEX_DIR = Path("/700gpart/chess/data/analysis/bse_neighbors_1800_2200_1m")


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def load_config(index_dir: Path) -> dict:
    config_path = index_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def vector_file(index_dir: Path, metric: Literal["cosine", "l2"]) -> Path:
    if metric == "cosine":
        return index_dir / "vectors_norm.fp16.npy"
    return index_dir / "vectors_raw.fp16.npy"


def output_file(index_dir: Path, metric: Literal["cosine", "l2"], index_type: str) -> Path:
    suffix = index_type.lower().replace("_", "")
    return index_dir / f"faiss_{metric}_{suffix}.index"


def faiss_metric(metric: Literal["cosine", "l2"]):
    import faiss

    return faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2


def make_index(d: int, metric: Literal["cosine", "l2"], args: argparse.Namespace):
    import faiss

    metric_type = faiss_metric(metric)
    if args.index_type == "flat":
        if metric == "cosine":
            return faiss.IndexFlatIP(d)
        return faiss.IndexFlatL2(d)
    if args.index_type == "ivfflat":
        quantizer = faiss.IndexFlatIP(d) if metric == "cosine" else faiss.IndexFlatL2(d)
        return faiss.IndexIVFFlat(quantizer, d, args.nlist, metric_type)
    if args.index_type == "ivfpq":
        if d % args.pq_m != 0:
            raise SystemExit(f"dimension {d} must be divisible by --pq-m {args.pq_m}")
        quantizer = faiss.IndexFlatIP(d) if metric == "cosine" else faiss.IndexFlatL2(d)
        return faiss.IndexIVFPQ(quantizer, d, args.nlist, args.pq_m, args.pq_nbits, metric_type)
    raise AssertionError(args.index_type)


def sample_training_vectors(vectors: np.ndarray, train_samples: int, seed: int) -> np.ndarray:
    n = vectors.shape[0]
    k = min(max(1, train_samples), n)
    rng = np.random.default_rng(seed)
    rows = rng.choice(n, size=k, replace=False)
    rows.sort()
    print(f"loading {k:,} training vectors into float32 ({sizeof_fmt(k * vectors.shape[1] * 4)})", flush=True)
    t0 = time.perf_counter()
    train = np.ascontiguousarray(vectors[rows], dtype=np.float32)
    print(f"loaded training sample in {time.perf_counter() - t0:.2f}s", flush=True)
    return train


def train_index(index, vectors: np.ndarray, args: argparse.Namespace) -> None:
    if index.is_trained:
        print("index does not require training", flush=True)
        return
    train = sample_training_vectors(vectors, args.train_samples, args.seed)
    print("training FAISS index", flush=True)
    t0 = time.perf_counter()
    index.train(train)
    print(f"trained in {time.perf_counter() - t0:.2f}s", flush=True)


def add_vectors(index, vectors: np.ndarray, batch_size: int) -> None:
    n = vectors.shape[0]
    print(f"adding {n:,} vectors in batches of {batch_size:,}", flush=True)
    t0 = time.perf_counter()
    for start in tqdm(range(0, n, batch_size), desc="adding vectors", unit="batch"):
        end = min(start + batch_size, n)
        batch = np.ascontiguousarray(vectors[start:end], dtype=np.float32)
        index.add(batch)
    elapsed = time.perf_counter() - t0
    print(f"added {n:,} vectors in {elapsed:.2f}s ({n / max(elapsed, 1e-12):,.0f} vec/s)", flush=True)


def set_runtime_params(index, args: argparse.Namespace) -> None:
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe


def sanity_search(index, vectors: np.ndarray, metric: Literal["cosine", "l2"], args: argparse.Namespace) -> None:
    if args.sanity_queries <= 0:
        return
    rng = np.random.default_rng(args.seed + 1)
    rows = rng.choice(vectors.shape[0], size=min(args.sanity_queries, vectors.shape[0]), replace=False)
    queries = np.ascontiguousarray(vectors[rows], dtype=np.float32)
    print(f"running sanity search: {len(rows):,} queries, k={args.sanity_k}", flush=True)
    t0 = time.perf_counter()
    distances, ids = index.search(queries, args.sanity_k)
    elapsed = time.perf_counter() - t0
    print(f"sanity search elapsed: {elapsed:.3f}s ({len(rows) / max(elapsed, 1e-12):,.1f} q/s)", flush=True)
    for i in range(min(5, len(rows))):
        payload = {
            "query_row": int(rows[i]),
            "ids": [int(x) for x in ids[i].tolist()],
            "scores_or_distances": [float(x) for x in distances[i].tolist()],
        }
        if metric == "l2":
            payload["rms"] = [math.sqrt(max(float(x), 0.0) / vectors.shape[1]) for x in distances[i].tolist()]
        print("  sanity", json.dumps(payload), flush=True)


def build_one(metric: Literal["cosine", "l2"], args: argparse.Namespace) -> None:
    import faiss

    t0 = time.perf_counter()
    in_path = vector_file(args.index_dir, metric)
    out_path = output_file(args.index_dir, metric, args.index_type)
    if out_path.exists() and not args.force:
        print(f"skipping existing {metric} index: {out_path}", flush=True)
        return
    if not in_path.exists():
        raise SystemExit(f"missing vector file: {in_path}")

    print("\n" + "=" * 80, flush=True)
    print(f"building {metric} FAISS index", flush=True)
    print(f"input:  {in_path} ({sizeof_fmt(in_path.stat().st_size)})", flush=True)
    print(f"output: {out_path}", flush=True)
    print("opening vector memmap", flush=True)
    vectors = np.load(in_path, mmap_mode="r", allow_pickle=False)
    print(f"vectors shape: {vectors.shape}; dtype={vectors.dtype}", flush=True)

    index = make_index(int(vectors.shape[1]), metric, args)
    set_runtime_params(index, args)
    print(f"index type: {type(index).__name__}", flush=True)
    print(f"index metric: {metric}; nlist={args.nlist}; nprobe={args.nprobe}; pq_m={args.pq_m}; pq_nbits={args.pq_nbits}", flush=True)

    train_index(index, vectors, args)
    add_vectors(index, vectors, args.add_batch_size)
    set_runtime_params(index, args)
    sanity_search(index, vectors, metric, args)

    print("writing FAISS index", flush=True)
    faiss.write_index(index, str(out_path))
    print(f"wrote {out_path} ({sizeof_fmt(out_path.stat().st_size)})", flush=True)
    print(f"{metric} total elapsed: {time.perf_counter() - t0:.2f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--metric", choices=["cosine", "l2", "both"], default="both")
    parser.add_argument("--index-type", choices=["flat", "ivfflat", "ivfpq"], default="ivfpq")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--pq-m", type=int, default=96)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--add-batch-size", type=int, default=4096)
    parser.add_argument("--sanity-queries", type=int, default=16)
    parser.add_argument("--sanity-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--threads", type=int, default=0, help="FAISS OpenMP threads; 0 leaves default")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("faiss is not installed. Run `uv sync` after adding faiss-cpu to pyproject.toml.") from exc

    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)
    print(f"index dir: {args.index_dir}", flush=True)
    config = load_config(args.index_dir)
    print(f"config kind: {config.get('kind')}", flush=True)
    print(f"bank shape: {config.get('bank_shape')} flat_dim={config.get('flat_dim')}", flush=True)
    print(f"FAISS version: {getattr(faiss, '__version__', 'unknown')}", flush=True)
    print(f"FAISS threads: {faiss.omp_get_max_threads()}", flush=True)

    metrics: list[Literal["cosine", "l2"]] = ["cosine", "l2"] if args.metric == "both" else [args.metric]  # type: ignore[list-item]
    for metric in metrics:
        build_one(metric, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
