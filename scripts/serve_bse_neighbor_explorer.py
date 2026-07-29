#!/usr/bin/env python3
"""Serve the BSE nearest-neighbor explorer UI and API.

The server loads a prebuilt index from scripts/build_bse_neighbor_index.py:
raw flattened fp16 banks and normalized flattened fp16 banks are loaded into RAM.
Search is deliberately brute-force and transparent; at 100k rows this is fine.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chessgm.tokenizer import VOCAB
from evaluate_board_state_q_probe import checkpoint_sort_key
from train_board_state_q_probe import QBoardStateProbeTransformer

try:
    import chess
    import chess.engine
    import chess.pgn
except ImportError as exc:  # pragma: no cover - startup config failure
    raise SystemExit("python-chess is required; install project dependencies first") from exc

DEFAULT_INDEX_DIR = ROOT / "data" / "analysis" / "bse_neighbors_1800_2200_100k"
DEFAULT_STATIC_DIR = ROOT / "tools" / "bse_neighbor_explorer"


class SearchRequest(BaseModel):
    game_id: int
    ply: int
    distance: Literal["l2", "cosine"] = "cosine"
    top_k: int = 5
    exclude_same_game: bool = True
    distinct_games: bool = True


class SampleRequest(BaseModel):
    mode: Literal["random", "bucket"] = "random"
    bucket: int | None = None
    terminal: bool | None = None


class AnalyzeRequest(BaseModel):
    game_id: int
    ply: int
    depth: int = 12
    movetime_ms: int = 250
    multipv: int = 5
    eval_played_move: bool = True


class AnalyzeManyRequest(BaseModel):
    positions: list[dict[str, int]]
    depth: int = 10
    movetime_ms: int = 150
    multipv: int = 5
    eval_played_move: bool = True


@dataclass
class ExplorerState:
    index_dir: Path
    checkpoint: Path
    device: str
    stockfish_path: Path | None
    config: dict[str, Any]
    metadata: np.ndarray
    vectors_raw: torch.Tensor
    vectors_norm: torch.Tensor
    moves: np.ndarray
    offsets: np.ndarray
    pgns: list[str]
    headers: list[dict[str, str]]
    model: QBoardStateProbeTransformer
    context_plies: int
    pad_id: int
    rng: random.Random


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
    if not candidates:
        raise SystemExit(f"No .pt checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def resolve_checkpoint(path: Path) -> Path:
    return latest_checkpoint(path) if path.is_dir() else path


def load_model(checkpoint_path: Path, device: str) -> tuple[QBoardStateProbeTransformer, dict[str, Any]]:
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
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_state(args: argparse.Namespace) -> ExplorerState:
    index_dir = args.index_dir
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = resolve_checkpoint(args.checkpoint or Path(config["checkpoint"]))
    print(f"BSE checkpoint: {checkpoint}")
    print(f"index dir: {index_dir}")
    model, ckpt = load_model(checkpoint, args.device)
    ckpt_args = ckpt.get("args", {})

    metadata = np.load(index_dir / "metadata.npy", allow_pickle=False)
    # Load fully into RAM, as requested. Keep fp16 storage; cast chunks to float32 during math.
    vectors_raw = torch.from_numpy(np.load(index_dir / "vectors_raw.fp16.npy", mmap_mode=None))
    vectors_norm = torch.from_numpy(np.load(index_dir / "vectors_norm.fp16.npy", mmap_mode=None))
    games_dir = index_dir / "games"
    moves = np.load(games_dir / "moves.npy", mmap_mode="r")
    offsets = np.load(games_dir / "offsets.npy", mmap_mode="r")
    pgn_rows = read_jsonl(games_dir / "pgn_texts.jsonl")
    header_rows = read_jsonl(games_dir / "game_headers.jsonl")
    pgns = [row["pgn"] for row in sorted(pgn_rows, key=lambda r: r["game_id"])]
    headers = [row["headers"] for row in sorted(header_rows, key=lambda r: r["game_id"])]
    return ExplorerState(
        index_dir=index_dir,
        checkpoint=checkpoint,
        device=args.device,
        stockfish_path=args.stockfish_path,
        config=config,
        metadata=metadata,
        vectors_raw=vectors_raw,
        vectors_norm=vectors_norm,
        moves=moves,
        offsets=offsets,
        pgns=pgns,
        headers=headers,
        model=model,
        context_plies=int(ckpt_args.get("context_plies", 128)),
        pad_id=int(ckpt_args.get("pad_id", 0)),
        rng=random.Random(args.seed),
    )


def strip_check_mate_tokens(prefix: np.ndarray, pad_id: int, ply_expr: int) -> np.ndarray:
    from chessgm.tokenizer import TOKEN_TO_ID

    leak = {TOKEN_TO_ID["CHECK"], TOKEN_TO_ID["MATE"]}
    stripped = np.full(prefix.shape, pad_id, dtype=prefix.dtype)
    for row_i, row in enumerate(prefix):
        kept = [int(tok) for tok in row if int(tok) not in leak]
        kept = kept[:ply_expr]
        stripped[row_i, : len(kept)] = kept
    return stripped


def make_prefix_x(state: ExplorerState, game_id: int, ply: int) -> np.ndarray:
    validate_game_ply(state, game_id, ply)
    start = int(state.offsets[game_id])
    prefix = state.moves[start : start + ply]
    prefix = strip_check_mate_tokens(prefix, pad_id=state.pad_id, ply_expr=int(state.moves.shape[1]))
    if len(prefix) >= state.context_plies:
        return prefix[-state.context_plies :].astype(np.int64, copy=False)
    pad = np.full((state.context_plies - len(prefix), state.moves.shape[1]), state.pad_id, dtype=np.uint16)
    return np.concatenate([pad, prefix], axis=0).astype(np.int64, copy=False)


def encode_query(state: ExplorerState, game_id: int, ply: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(make_prefix_x(state, game_id, ply))[None, ...].to(state.device)
    with torch.inference_mode():
        flat = state.model.encoder(x).detach().float().reshape(-1).cpu()
    norm = flat / max(float(torch.linalg.vector_norm(flat)), 1e-12)
    return flat, norm


def validate_game_ply(state: ExplorerState, game_id: int, ply: int) -> None:
    if game_id < 0 or game_id >= len(state.offsets) - 1:
        raise HTTPException(404, f"unknown game_id={game_id}")
    length = int(state.offsets[game_id + 1] - state.offsets[game_id])
    if ply < 1 or ply > length:
        raise HTTPException(400, f"ply must be in [1, {length}], got {ply}")


def game_at_ply(state: ExplorerState, game_id: int, ply: int) -> tuple[chess.Board, list[str], list[str]]:
    validate_game_ply(state, game_id, ply)
    game = chess.pgn.read_game(io.StringIO(state.pgns[game_id]))
    if game is None:
        raise HTTPException(500, f"could not parse stored PGN for game {game_id}")
    board = game.board()
    san_played: list[str] = []
    uci_played: list[str] = []
    for i, move in enumerate(game.mainline_moves(), start=1):
        if i > ply:
            break
        san_played.append(board.san(move))
        uci_played.append(move.uci())
        board.push(move)
    return board, san_played, uci_played


def board_payload(state: ExplorerState, game_id: int, ply: int) -> dict[str, Any]:
    board, san, uci = game_at_ply(state, game_id, ply)
    length = int(state.offsets[game_id + 1] - state.offsets[game_id])
    headers = state.headers[game_id]
    return {
        "game_id": game_id,
        "ply": ply,
        "terminal_ply": length,
        "headers": headers,
        "fen": board.fen(),
        "board_fen": board.board_fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "next_to_play": "black" if ply % 2 == 1 else "white",
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "legal_move_count": board.legal_moves.count(),
        "last_move_san": san[-1] if san else None,
        "last_move_uci": uci[-1] if uci else None,
        "next_move_san": next_move_san(state, game_id, ply),
        "san": san,
        "uci": uci,
    }


def next_move(state: ExplorerState, game_id: int, ply: int) -> tuple[str | None, str | None]:
    length = int(state.offsets[game_id + 1] - state.offsets[game_id])
    if ply >= length:
        return None, None
    game = chess.pgn.read_game(io.StringIO(state.pgns[game_id]))
    if game is None:
        return None, None
    board = game.board()
    for i, move in enumerate(game.mainline_moves(), start=1):
        if i == ply + 1:
            return board.san(move), move.uci()
        board.push(move)
    return None, None


def next_move_san(state: ExplorerState, game_id: int, ply: int) -> str | None:
    return next_move(state, game_id, ply)[0]


def brute_force_search(state: ExplorerState, req: SearchRequest) -> list[dict[str, Any]]:
    q_raw, q_norm = encode_query(state, req.game_id, req.ply)
    top_k_scan = min(max(req.top_k * 200, 500), len(state.metadata))
    chunk_size = 8192
    candidates: list[tuple[float, int]] = []
    largest = req.distance == "cosine"
    score_name = "cosine" if largest else "l2"
    query = q_norm.float() if largest else q_raw.float()
    matrix = state.vectors_norm if largest else state.vectors_raw

    for start in range(0, len(state.metadata), chunk_size):
        chunk = matrix[start : start + chunk_size].float()
        if largest:
            scores = chunk @ query
        else:
            scores = torch.sum((chunk - query) ** 2, dim=1)
        k = min(top_k_scan, scores.numel())
        vals, idxs = torch.topk(scores, k=k, largest=largest)
        candidates.extend((float(v), int(start + i)) for v, i in zip(vals.cpu().tolist(), idxs.cpu().tolist(), strict=True))

    candidates.sort(key=lambda x: x[0], reverse=largest)
    out: list[dict[str, Any]] = []
    seen_games: set[int] = set()
    for score, row_i in candidates:
        meta = state.metadata[row_i]
        gid = int(meta["game_id"])
        if req.exclude_same_game and gid == req.game_id:
            continue
        if req.distinct_games and gid in seen_games:
            continue
        seen_games.add(gid)
        ply = int(meta["ply"])
        out.append(
            {
                "rank": len(out) + 1,
                "row": int(row_i),
                "game_id": gid,
                "ply": ply,
                "is_terminal": bool(meta["is_terminal"]),
                score_name: score,
                "position": board_payload(state, gid, ply),
            }
        )
        if len(out) >= req.top_k:
            break
    return out


def material_summary(board: chess.Board) -> dict[str, Any]:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    counts = {"white": defaultdict(int), "black": defaultdict(int)}
    score = {"white": 0, "black": 0}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        side = "white" if piece.color == chess.WHITE else "black"
        counts[side][names[piece.piece_type]] += 1
        score[side] += values[piece.piece_type]
    return {
        "counts": {k: dict(v) for k, v in counts.items()},
        "score": score,
        "balance_white_minus_black": score["white"] - score["black"],
    }


def stockfish_eval(
    state: ExplorerState,
    board: chess.Board,
    depth: int,
    movetime_ms: int,
    *,
    multipv: int = 5,
    played_next_uci: str | None = None,
    eval_played_move: bool = True,
) -> dict[str, Any]:
    if state.stockfish_path is None or not state.stockfish_path.exists():
        return {"available": False, "reason": "stockfish path not configured or missing"}
    try:
        with chess.engine.SimpleEngine.popen_uci(str(state.stockfish_path)) as engine:
            limit = chess.engine.Limit(depth=max(1, depth), time=max(1, movetime_ms) / 1000.0)
            raw = engine.analyse(board, limit, multipv=max(1, min(int(multipv), 10)))
            infos = raw if isinstance(raw, list) else [raw]
            top_moves = []
            for info in infos:
                score = info["score"].white()
                pv = info.get("pv", [])
                first = pv[0] if pv else None
                top_moves.append(
                    {
                        "rank": int(info.get("multipv", len(top_moves) + 1)),
                        "move_uci": first.uci() if first else None,
                        "move_san": board.san(first) if first else None,
                        "score_cp": score.score(mate_score=100000),
                        "mate": score.mate(),
                        "depth": info.get("depth"),
                        "pv_uci": [m.uci() for m in pv],
                    }
                )
            top_moves.sort(key=lambda x: x["rank"])
            played_rank = None
            for row in top_moves:
                if played_next_uci and row["move_uci"] == played_next_uci:
                    played_rank = row["rank"]
                    break

            played_after = None
            eval_swing_cp = None
            if eval_played_move and played_next_uci:
                try:
                    move = chess.Move.from_uci(played_next_uci)
                    if move in board.legal_moves:
                        next_board = board.copy(stack=False)
                        next_board.push(move)
                        next_info = engine.analyse(next_board, limit)
                        next_score = next_info["score"].white()
                        played_after = {
                            "score_cp": next_score.score(mate_score=100000),
                            "mate": next_score.mate(),
                            "depth": next_info.get("depth"),
                        }
                        if top_moves and top_moves[0]["score_cp"] is not None and played_after["score_cp"] is not None:
                            eval_swing_cp = int(played_after["score_cp"] - top_moves[0]["score_cp"])
                except Exception as exc:
                    played_after = {"error": str(exc)}

            return {
                "available": True,
                "best": top_moves[0] if top_moves else None,
                "top_moves": top_moves,
                "played_next_uci": played_next_uci,
                "played_move_rank": played_rank,
                "played_after": played_after,
                "eval_swing_cp_vs_best": eval_swing_cp,
            }
    except Exception as exc:  # pragma: no cover - external engine
        return {"available": False, "reason": str(exc)}


def create_app(state: ExplorerState, static_dir: Path) -> FastAPI:
    app = FastAPI(title="BSE neighbor explorer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        lengths = np.diff(state.offsets)
        max_bucket = int((int(lengths.max()) - 1) // int(state.config.get("bucket_plies", 5))) if len(lengths) else 0
        return {
            "index_dir": str(state.index_dir),
            "checkpoint": str(state.checkpoint),
            "num_positions": int(len(state.metadata)),
            "num_games": int(len(state.offsets) - 1),
            "bucket_plies": int(state.config.get("bucket_plies", 5)),
            "max_bucket": max_bucket,
            "bank_shape": state.config.get("bank_shape"),
            "stockfish_configured": state.stockfish_path is not None and state.stockfish_path.exists(),
        }

    @app.post("/api/sample")
    def api_sample(req: SampleRequest) -> dict[str, Any]:
        rows = state.metadata
        mask = np.ones(len(rows), dtype=bool)
        if req.terminal is not None:
            mask &= rows["is_terminal"] == bool(req.terminal)
        if req.mode == "bucket":
            if req.bucket is None:
                raise HTTPException(400, "bucket mode requires bucket")
            bucket_plies = int(state.config.get("bucket_plies", 5))
            mask &= ((rows["ply"].astype(np.int64) - 1) // bucket_plies) == int(req.bucket)
        candidates = np.flatnonzero(mask)
        if len(candidates) == 0:
            raise HTTPException(404, "no indexed positions match sample constraints")
        row_i = int(candidates[state.rng.randrange(len(candidates))])
        meta = rows[row_i]
        return {"row": row_i, "position": board_payload(state, int(meta["game_id"]), int(meta["ply"]))}

    @app.get("/api/game/{game_id}")
    def api_game(game_id: int) -> dict[str, Any]:
        if game_id < 0 or game_id >= len(state.offsets) - 1:
            raise HTTPException(404, f"unknown game_id={game_id}")
        return {
            "game_id": game_id,
            "terminal_ply": int(state.offsets[game_id + 1] - state.offsets[game_id]),
            "headers": state.headers[game_id],
            "pgn": state.pgns[game_id],
        }

    @app.get("/api/position")
    def api_position(game_id: int = Query(...), ply: int = Query(...)) -> dict[str, Any]:
        return board_payload(state, game_id, ply)

    @app.post("/api/search")
    def api_search(req: SearchRequest) -> dict[str, Any]:
        req.top_k = min(max(req.top_k, 1), 20)
        return {"query": board_payload(state, req.game_id, req.ply), "results": brute_force_search(state, req)}

    def analyze_position(req: AnalyzeRequest) -> dict[str, Any]:
        board, san, uci = game_at_ply(state, req.game_id, req.ply)
        captures = [board.san(m) for m in board.legal_moves if board.is_capture(m)]
        checks = []
        for move in board.legal_moves:
            b = board.copy(stack=False)
            b.push(move)
            if b.is_check():
                checks.append(board.san(move))
        _, next_uci = next_move(state, req.game_id, req.ply)
        return {
            "position": board_payload(state, req.game_id, req.ply),
            "material": material_summary(board),
            "legal_moves": [board.san(m) for m in board.legal_moves],
            "captures": captures,
            "checks": checks,
            "stockfish": stockfish_eval(
                state,
                board,
                req.depth,
                req.movetime_ms,
                multipv=req.multipv,
                played_next_uci=next_uci,
                eval_played_move=req.eval_played_move,
            ),
        }

    @app.post("/api/analyze")
    def api_analyze(req: AnalyzeRequest) -> dict[str, Any]:
        return analyze_position(req)

    @app.post("/api/analyze_many")
    def api_analyze_many(req: AnalyzeManyRequest) -> dict[str, Any]:
        rows = []
        for pos in req.positions[:20]:
            rows.append(
                analyze_position(
                    AnalyzeRequest(
                        game_id=int(pos["game_id"]),
                        ply=int(pos["ply"]),
                        depth=req.depth,
                        movetime_ms=req.movetime_ms,
                        multipv=req.multipv,
                        eval_played_move=req.eval_played_move,
                    )
                )
            )
        return {"analyses": rows}

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override BSE checkpoint; file or directory")
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--stockfish-path", type=Path, default=ROOT / "tools" / "stockfish" / "stockfish")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    state = load_state(args)
    app = create_app(state, args.static_dir)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
