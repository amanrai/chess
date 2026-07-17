#!/usr/bin/env python3
"""Build aligned board-state targets and a stratified probe-sampling plan.

This is a fresh preprocessing stage for the board-state verifier.  It replays
exactly the accepted PGNs used by the verifier game store and writes one packed
64-square board target after every stored ply.  It also materializes a compact
``probe_samples.npy`` plan of ``(game_index, prefix_plies)`` pairs.  The plan
chooses a prefix-position bucket first, then samples uniformly from games that
reach it, so common early positions are sampled broadly without over-repeating
rare long games.

Outputs (under ``--out-dir``):
  board_after_packed.npy  uint8[total_plies, 32], two 4-bit labels per byte
  probe_samples.npy       uint32[num_samples, 2], (game_index, prefix_plies)
  manifest.json

Occupant labels are EMPTY=0; white pawn..king=1..6; black pawn..king=7..12.
Square order is python-chess's a1 through h8 order.  Input/store alignment is
verified packet-by-packet, so this fails rather than silently writing targets
for a differently filtered or ordered PGN corpus.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessgm.tokenizer import ChessTokenizer  # noqa: E402
from preprocess_verifier_dataset import (  # noqa: E402
    DEFAULT_INPUTS,
    RESULT_TO_LABEL,
    iter_pgn_games,
    pgn_game_to_packets,
    result_from_pgn_text,
)

DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "lumbras" / "verifier"


# These intentionally match python-chess's PAWN..KING values (1..6).
EMPTY = 0
WHITE_OFFSET = 0
BLACK_OFFSET = 6
PACKED_SQUARES = 32
_WORKER_TOKENIZER: ChessTokenizer | None = None
_WORKER_CHESS = None


def pack_board_after(board) -> np.ndarray:
    """Pack a python-chess board's a1..h8 occupant labels into 32 bytes."""
    labels = np.zeros(64, dtype=np.uint8)
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue
        # python-chess: WHITE=True, BLACK=False; piece_type is PAWN..KING (1..6).
        labels[square] = int(piece.piece_type) + (WHITE_OFFSET if piece.color else BLACK_OFFSET)
    return labels[0::2] | (labels[1::2] << 4)


def _require_python_chess():
    try:
        import chess
        import chess.pgn
    except ImportError as exc:
        raise SystemExit(
            "python-chess is required for board replay. Install project dependencies "
            "(for example: uv sync or pip install -r requirements.txt)."
        ) from exc
    return chess


def replay_packed_boards(game_text: str, chess) -> list[np.ndarray]:
    """Return packed post-ply boards from the PGN main line."""
    game = chess.pgn.read_game(io.StringIO(game_text))
    if game is None:
        raise ValueError("python-chess could not parse PGN game")
    if game.errors:
        raise ValueError(f"python-chess PGN errors: {game.errors[0]}")
    board = game.board()
    packed: list[np.ndarray] = []
    for move in game.mainline_moves():
        board.push(move)
        packed.append(pack_board_after(board))
    return packed


def replay_game_for_store(
    game_text: str, seq_len: int, tokenizer: ChessTokenizer, chess
) -> tuple[int | None, np.ndarray | None, np.ndarray | None]:
    """Return accepted game's label, packets, and post-ply packed boards."""
    result = result_from_pgn_text(game_text)
    if result not in RESULT_TO_LABEL:
        return None, None, None
    packets = pgn_game_to_packets(game_text, tokenizer, seq_len)
    if packets is None or len(packets) < 2:
        return None, None, None
    boards = replay_packed_boards(game_text, chess)
    if len(boards) != len(packets):
        raise ValueError(
            f"board replay produced {len(boards)} plies but tokenization produced {len(packets)}"
        )
    return RESULT_TO_LABEL[result], packets, np.asarray(boards, dtype=np.uint8)


def _init_replay_worker() -> None:
    global _WORKER_CHESS, _WORKER_TOKENIZER
    _WORKER_TOKENIZER = ChessTokenizer()
    _WORKER_CHESS = _require_python_chess()


def _replay_game_worker(args: tuple[str, int]) -> tuple[int | None, np.ndarray | None, np.ndarray | None]:
    game_text, seq_len = args
    tokenizer = _WORKER_TOKENIZER or ChessTokenizer()
    chess = _WORKER_CHESS or _require_python_chess()
    return replay_game_for_store(game_text, seq_len, tokenizer, chess)


def bucket_plan(
    lengths: np.ndarray,
    samples: int,
    bucket_plies: int,
    max_prefix_plies: int,
    allocation_alpha: float,
    min_samples_per_bucket: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int | float]]]:
    """Build ``(game, prefix)`` samples using eligible-game bucket weighting.

    A bucket starting at ply ``s`` is eligible for every game of length >= s.
    Its allocation is proportional to ``eligible_games ** allocation_alpha``.
    Sampling a game uniformly from that eligible set preserves the corpus's
    natural final-length mix at that prefix position.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if bucket_plies < 1:
        raise ValueError("bucket_plies must be >= 1")
    if allocation_alpha <= 0:
        raise ValueError("allocation_alpha must be > 0")
    if min_samples_per_bucket < 0:
        raise ValueError("min_samples_per_bucket must be >= 0")

    max_available = int(lengths.max()) if len(lengths) else 0
    max_prefix = min(max_available, max_prefix_plies) if max_prefix_plies else max_available
    if max_prefix < 1:
        raise ValueError("no positive-length games available for board-state sampling")
    starts = np.arange(1, max_prefix + 1, bucket_plies, dtype=np.int64)
    ends = np.minimum(starts + bucket_plies - 1, max_prefix)
    sorted_lengths = np.sort(lengths)
    eligible_counts = len(lengths) - np.searchsorted(sorted_lengths, starts, side="left")
    active = eligible_counts > 0
    starts, ends, eligible_counts = starts[active], ends[active], eligible_counts[active]
    if min_samples_per_bucket * len(starts) > samples:
        raise ValueError("samples is smaller than min_samples_per_bucket times active buckets")

    weights = eligible_counts.astype(np.float64) ** allocation_alpha
    remaining = samples - min_samples_per_bucket * len(starts)
    exact = remaining * weights / weights.sum()
    allocations = np.floor(exact).astype(np.int64) + min_samples_per_bucket
    # Largest-remainder rounding keeps the requested total exact and reproducible.
    remainder = remaining - int(np.floor(exact).sum())
    if remainder:
        winners = np.argsort(-(exact - np.floor(exact)), kind="stable")[:remainder]
        allocations[winners] += 1

    plan = np.empty((samples, 2), dtype=np.uint32)
    rng = np.random.default_rng(seed)
    cursor = 0
    records: list[dict[str, int | float]] = []
    for start, end, eligible, allocation in zip(starts, ends, eligible_counts, allocations, strict=True):
        game_indices = np.flatnonzero(lengths >= start)
        chosen = game_indices[rng.integers(len(game_indices), size=int(allocation))]
        chosen_lengths = lengths[chosen]
        prefix_hi = np.minimum(chosen_lengths, end)
        prefix_lengths = rng.integers(start, prefix_hi + 1)
        next_cursor = cursor + int(allocation)
        plan[cursor:next_cursor, 0] = chosen
        plan[cursor:next_cursor, 1] = prefix_lengths
        records.append(
            {
                "plies_lo": int(start),
                "plies_hi": int(end),
                "eligible_games": int(eligible),
                "samples": int(allocation),
                "sample_percent": 100.0 * int(allocation) / samples,
            }
        )
        cursor = next_cursor
    assert cursor == samples
    rng.shuffle(plan)
    return plan, records


def build_board_targets(
    inputs: list[Path],
    moves: np.ndarray,
    offsets: np.ndarray,
    results: np.ndarray,
    output_path: Path,
    progress: bool,
    workers: int,
    chunksize: int,
) -> None:
    """Replay inputs and write targets, proving each accepted game matches store rows.

    Worker processes parse/tokenize/replay games in parallel. ``Executor.map``
    preserves source order, while this parent process remains the sole writer
    and performs the game-store alignment checks before each write.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1")
    _require_python_chess()
    target = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.uint8, shape=(len(moves), PACKED_SQUARES)
    )
    game_i = 0
    try:
        for path in inputs:
            game_args = ((game_text, int(moves.shape[1])) for game_text in iter_pgn_games(path))
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_replay_worker) as pool:
                    processed = pool.map(_replay_game_worker, game_args, chunksize=chunksize)
                    game_i = _write_replayed_games(
                        processed, path, target, moves, offsets, results, game_i, progress
                    )
            else:
                processed = (_replay_game_worker(item) for item in game_args)
                game_i = _write_replayed_games(
                    processed, path, target, moves, offsets, results, game_i, progress
                )
        if game_i != len(results):
            raise ValueError(f"inputs yielded {game_i} accepted games but verifier store contains {len(results)}")
    except Exception:
        del target
        output_path.unlink(missing_ok=True)
        raise
    target.flush()
    del target


def _write_replayed_games(processed, path, target, moves, offsets, results, game_i: int, progress: bool) -> int:
    """Validate ordered replay results and write them, returning the game index."""
    iterator = tqdm(processed, desc=f"replaying {path.name}", unit="game", disable=not progress)
    for label, packets, boards in iterator:
        if packets is None:
            continue
        assert label is not None and boards is not None
        if game_i >= len(results):
            raise ValueError("inputs contain more accepted games than the verifier store")
        start, end = int(offsets[game_i]), int(offsets[game_i + 1])
        if int(results[game_i]) != label:
            raise ValueError(f"result mismatch at verifier game {game_i}")
        if end - start != len(packets) or not np.array_equal(moves[start:end], packets):
            raise ValueError(
                f"move-store mismatch at verifier game {game_i}; inputs must be the exact "
                "PGNs and ordering used to create this verifier store"
            )
        target[start:end] = boards
        game_i += 1
        iterator.set_postfix(accepted=f"{game_i:,}")
    return game_i


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Existing verifier game store")
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <data-dir>/board_state")
    parser.add_argument("--samples", type=int, default=20_000_000, help="Board-state probe positions to materialize")
    parser.add_argument("--bucket-plies", type=int, default=10)
    parser.add_argument("--max-prefix-plies", type=int, default=250, help="0 includes every stored ply")
    parser.add_argument("--allocation-alpha", type=float, default=1.15, help="Eligible-game allocation exponent; >1 favors early buckets")
    parser.add_argument("--min-samples-per-bucket", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="PGN replay worker processes; use $(nproc)")
    parser.add_argument("--chunksize", type=int, default=32, help="Games per worker task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    missing = [path for path in args.inputs if not path.exists()]
    if missing:
        raise SystemExit("Missing input(s): " + ", ".join(map(str, missing)))
    for name in ("moves.npy", "offsets.npy", "results.npy"):
        if not (args.data_dir / name).exists():
            raise SystemExit(f"Missing verifier-store file: {args.data_dir / name}")

    moves = np.load(args.data_dir / "moves.npy", mmap_mode="r")
    offsets = np.load(args.data_dir / "offsets.npy", mmap_mode="r")
    results = np.load(args.data_dir / "results.npy", mmap_mode="r")
    if len(offsets) != len(results) + 1 or int(offsets[-1]) != len(moves):
        raise SystemExit("Verifier store arrays are inconsistent")

    workers = max(1, args.workers)
    if workers > (os.cpu_count() or workers) and not args.no_progress:
        print(f"warning: workers={workers} exceeds detected cpu_count={os.cpu_count()}")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")

    out_dir = args.out_dir or args.data_dir / "board_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    board_path = out_dir / "board_after_packed.npy"
    print(f"writing aligned board targets: {board_path}")
    build_board_targets(
        args.inputs, moves, offsets, results, board_path, progress=not args.no_progress,
        workers=workers, chunksize=args.chunksize,
    )

    plan, buckets = bucket_plan(
        np.diff(offsets), args.samples, args.bucket_plies, args.max_prefix_plies,
        args.allocation_alpha, args.min_samples_per_bucket, args.seed,
    )
    plan_path = out_dir / "probe_samples.npy"
    np.save(plan_path, plan)
    manifest = {
        "kind": "board_state_verifier",
        "source_verifier_dir": str(args.data_dir),
        "inputs": [str(path) for path in args.inputs],
        "board_after_packed_path": str(board_path),
        "board_after_packed_shape": [int(len(moves)), PACKED_SQUARES],
        "board_after_packed_dtype": "uint8",
        "square_order": "a1 through h8 (python-chess square order)",
        "occupant_labels": {"EMPTY": 0, **{f"WHITE_{name}": i for i, name in enumerate(["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"], 1)}, **{f"BLACK_{name}": i for i, name in enumerate(["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"], 7)}},
        "probe_samples_path": str(plan_path),
        "probe_samples_shape": [int(len(plan)), 2],
        "probe_samples_dtype": "uint32",
        "probe_samples_columns": ["game_index", "prefix_plies"],
        "sampling": {
            "bucket_plies": args.bucket_plies,
            "max_prefix_plies": args.max_prefix_plies,
            "allocation_alpha": args.allocation_alpha,
            "min_samples_per_bucket": args.min_samples_per_bucket,
            "workers": workers,
            "chunksize": args.chunksize,
            "seed": args.seed,
            "buckets": buckets,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(plan):,} stratified probe samples: {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
