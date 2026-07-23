#!/usr/bin/env python3
"""Sample an Elo-heldout board-state evaluation set.

Default bucket is games where both players have Elo in [2000, 2200), which is
below the current 2200+ training corpus. The script streams Lumbras .7z PGN
archives, reservoir-samples accepted games, tokenizes their mainline move
prefixes with the project tokenizer, replays the board after every ply with
python-chess, and writes compact arrays for later all-64 board reconstruction
evaluation.

Outputs under --out-dir:
  games.pgn                 sampled PGNs, in stored order
  moves.npy                 uint16[total_plies, seq_len]
  offsets.npy               int64[num_games + 1]
  board_after_packed.npy    uint8[total_plies, 32], a1..h8 two nibbles/byte
  manifest.json
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chessgm.tokenizer import ChessTokenizer, VOCAB  # noqa: E402
from extract_lumbras_2200_splits import iter_games, parse_elo  # noqa: E402
from preprocess_board_state_verifier import pack_board_after  # noqa: E402
from preprocess_verifier_dataset import pgn_game_to_packets, result_from_pgn_text, RESULT_TO_LABEL  # noqa: E402

DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "lumbras" / "otb"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "lumbras" / "eval_board_state_2000_2199"


def require_python_chess():
    try:
        import chess
        import chess.pgn
    except ImportError as exc:
        raise SystemExit("python-chess is required; install requirements.txt first") from exc
    return chess


def in_elo_bucket(headers: dict[str, str], min_elo: int, max_elo: int) -> bool:
    white = parse_elo(headers.get("WhiteElo"))
    black = parse_elo(headers.get("BlackElo"))
    if white is None or black is None:
        return False
    floor = min(white, black)
    return min_elo <= floor < max_elo


def replay_packed_boards(game_text: str, chess) -> np.ndarray:
    game = chess.pgn.read_game(io.StringIO(game_text))
    if game is None:
        raise ValueError("python-chess could not parse PGN")
    if game.errors:
        raise ValueError(f"python-chess PGN errors: {game.errors[0]}")
    board = game.board()
    boards: list[np.ndarray] = []
    for move in game.mainline_moves():
        board.push(move)
        boards.append(pack_board_after(board))
    return np.asarray(boards, dtype=np.uint8)


def build_record(game_text: str, tokenizer: ChessTokenizer, seq_len: int, chess) -> tuple[np.ndarray, np.ndarray, int]:
    result = result_from_pgn_text(game_text)
    if result not in RESULT_TO_LABEL:
        raise ValueError(f"unsupported result: {result}")
    packets = pgn_game_to_packets(game_text, tokenizer, seq_len)
    if packets is None or len(packets) < 2:
        raise ValueError("empty or bad tokenized game")
    boards = replay_packed_boards(game_text, chess)
    if len(boards) != len(packets):
        raise ValueError(f"token/board ply mismatch: packets={len(packets)} boards={len(boards)}")
    return packets, boards, RESULT_TO_LABEL[result]


def reservoir_sample_games(
    archives: list[Path],
    *,
    samples: int,
    min_elo: int,
    max_elo: int,
    seed: int,
    progress: bool,
) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    stats = {
        "archives": [str(path) for path in archives],
        "seen_games": 0,
        "bucket_candidates": 0,
        "per_archive": {},
    }
    outer = tqdm(archives, desc="archives", unit="archive", disable=not progress)
    for archive in outer:
        local = {"seen_games": 0, "bucket_candidates": 0}
        for headers, game_text in iter_games(archive):
            stats["seen_games"] += 1
            local["seen_games"] += 1
            if not in_elo_bucket(headers, min_elo, max_elo):
                continue
            stats["bucket_candidates"] += 1
            local["bucket_candidates"] += 1
            seen = int(stats["bucket_candidates"])
            if len(reservoir) < samples:
                reservoir.append(game_text)
            else:
                j = rng.randrange(seen)
                if j < samples:
                    reservoir[j] = game_text
        stats["per_archive"][archive.name] = local
        outer.set_postfix(candidates=stats["bucket_candidates"], sampled=len(reservoir))
    return reservoir, stats


def write_eval_store(
    games: list[str],
    out_dir: Path,
    *,
    seq_len: int,
    min_elo: int,
    max_elo: int,
    seed: int,
    scan_stats: dict,
    progress: bool,
) -> dict:
    tokenizer = ChessTokenizer()
    chess = require_python_chess()
    accepted_games: list[str] = []
    moves_parts: list[np.ndarray] = []
    board_parts: list[np.ndarray] = []
    offsets = [0]
    results: list[int] = []
    skipped: dict[str, int] = {}

    for game_text in tqdm(games, desc="tokenizing/replaying sample", unit="game", disable=not progress):
        try:
            packets, boards, result = build_record(game_text, tokenizer, seq_len, chess)
        except Exception as exc:  # keep eval construction robust to rare malformed PGNs
            key = type(exc).__name__
            skipped[key] = skipped.get(key, 0) + 1
            continue
        accepted_games.append(game_text.rstrip() + "\n")
        moves_parts.append(packets)
        board_parts.append(boards)
        results.append(result)
        offsets.append(offsets[-1] + len(packets))

    if not accepted_games:
        raise SystemExit("No sampled games survived tokenization/replay")

    moves = np.concatenate(moves_parts, axis=0).astype(np.uint16, copy=False)
    boards = np.concatenate(board_parts, axis=0).astype(np.uint8, copy=False)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    results_arr = np.asarray(results, dtype=np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "games.pgn").write_text("\n".join(accepted_games) + "\n", encoding="utf-8")
    np.save(out_dir / "moves.npy", moves)
    np.save(out_dir / "offsets.npy", offsets_arr)
    np.save(out_dir / "results.npy", results_arr)
    np.save(out_dir / "board_after_packed.npy", boards)

    lengths = np.diff(offsets_arr)
    manifest = {
        "kind": "board_state_eval_bucket",
        "source": "Lumbra's Gigabase OTB PGN archives",
        "elo_bucket": {"definition": "both players' minimum Elo is in [min_elo, max_elo)", "min_elo": min_elo, "max_elo": max_elo},
        "seed": seed,
        "requested_samples": len(games),
        "accepted_games": int(len(accepted_games)),
        "total_plies": int(len(moves)),
        "seq_len": seq_len,
        "moves_path": str(out_dir / "moves.npy"),
        "offsets_path": str(out_dir / "offsets.npy"),
        "results_path": str(out_dir / "results.npy"),
        "board_after_packed_path": str(out_dir / "board_after_packed.npy"),
        "games_pgn_path": str(out_dir / "games.pgn"),
        "moves_shape": list(moves.shape),
        "offsets_shape": list(offsets_arr.shape),
        "results_shape": list(results_arr.shape),
        "board_after_packed_shape": list(boards.shape),
        "dtype": {"moves": "uint16", "offsets": "int64", "results": "int64", "board_after_packed": "uint8"},
        "square_order": "a1 through h8 (python-chess square order)",
        "occupant_labels": {"EMPTY": 0, "WHITE_PAWN": 1, "WHITE_KNIGHT": 2, "WHITE_BISHOP": 3, "WHITE_ROOK": 4, "WHITE_QUEEN": 5, "WHITE_KING": 6, "BLACK_PAWN": 7, "BLACK_KNIGHT": 8, "BLACK_BISHOP": 9, "BLACK_ROOK": 10, "BLACK_QUEEN": 11, "BLACK_KING": 12},
        "vocab_size": len(VOCAB),
        "vocab": VOCAB,
        "skipped_after_sampling": skipped,
        "scan_stats": scan_stats,
        "game_length_plies": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p10": float(np.percentile(lengths, 10)),
            "p90": float(np.percentile(lengths, 90)),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path, help="Lumbras .7z archives; default: data/raw/lumbras/otb/*.7z")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument("--max-elo", type=int, default=2200)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.min_elo >= args.max_elo:
        raise SystemExit("--min-elo must be < --max-elo")
    archives = args.archives or sorted(args.raw_dir.glob("*.7z"))
    if not archives:
        raise SystemExit(f"No archives found in {args.raw_dir}")
    missing = [path for path in archives if not path.exists()]
    if missing:
        raise SystemExit("Missing archive(s): " + ", ".join(map(str, missing)))

    games, scan_stats = reservoir_sample_games(
        archives,
        samples=args.samples,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        seed=args.seed,
        progress=not args.no_progress,
    )
    if len(games) < args.samples:
        print(f"warning: requested {args.samples} games but only found {len(games)} candidates")
    manifest = write_eval_store(
        games,
        args.out_dir,
        seq_len=args.seq_len,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        seed=args.seed,
        scan_stats=scan_stats,
        progress=not args.no_progress,
    )
    print("wrote board-state eval bucket:", args.out_dir)
    print(json.dumps({k: manifest[k] for k in ["accepted_games", "total_plies", "moves_shape", "board_after_packed_shape", "game_length_plies"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
