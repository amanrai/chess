#!/usr/bin/env python3
"""Serve a small read-only browser UI for the Lumbras metadata SQLite DB."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "tools" / "lumbras_metadata_store"
DEFAULT_DB = Path("/700gpart/chess/data/catalog/lumbras_otb/lumbras_otb_catalog.sqlite")

READONLY_PREFIXES = ("select", "with", "pragma")
BLOCKED_SQL = ("insert", "update", "delete", "drop", "alter", "create", "attach", "detach", "replace", "vacuum")


class QueryRequest(BaseModel):
    sql: str
    limit: int = 200


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"SQLite DB not found: {db_path}")
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def validate_readonly_sql(sql: str) -> str:
    stripped = sql.strip().rstrip(";")
    lowered = stripped.lower()
    if not stripped:
        raise HTTPException(status_code=400, detail="SQL is empty")
    if ";" in stripped:
        raise HTTPException(status_code=400, detail="Only one SQL statement is allowed")
    if not lowered.startswith(READONLY_PREFIXES):
        raise HTTPException(status_code=400, detail="Only read-only SELECT/WITH/PRAGMA statements are allowed")
    tokens = set(lowered.replace("(", " ").replace(")", " ").replace(",", " ").split())
    blocked = sorted(tokens.intersection(BLOCKED_SQL))
    if blocked:
        raise HTTPException(status_code=400, detail=f"Blocked SQL token(s): {', '.join(blocked)}")
    return stripped


def rows_to_payload(cursor: sqlite3.Cursor, rows: list[sqlite3.Row]) -> dict[str, Any]:
    columns = [desc[0] for desc in cursor.description or []]
    return {"columns": columns, "rows": [[row[col] for col in columns] for row in rows], "row_count": len(rows)}


def make_app(db_path: Path) -> FastAPI:
    app = FastAPI(title="Lumbras Metadata Store", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "db": str(db_path), "exists": db_path.exists()}

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        with connect(db_path) as conn:
            def scalar(sql: str) -> int:
                return int(conn.execute(sql).fetchone()[0])

            counts = {
                "games": scalar("select count(*) from games"),
                "headers": scalar("select count(*) from game_headers"),
                "both_1600_1800": scalar("select count(*) from games where min_elo >= 1600 and max_elo < 1800"),
                "both_1800_2200": scalar("select count(*) from games where min_elo >= 1800 and max_elo <= 2200"),
                "both_2200_plus": scalar("select count(*) from games where min_elo >= 2200"),
            }
            archives = [dict(row) for row in conn.execute(
                """
                select source_archive, count(*) as games
                from games
                group by source_archive
                order by source_archive
                """
            ).fetchall()]
            results = [dict(row) for row in conn.execute(
                """
                select coalesce(result, '(missing)') as result, count(*) as games
                from games
                group by result
                order by games desc
                """
            ).fetchall()]
            header_keys = [dict(row) for row in conn.execute(
                """
                select key, count(*) as n
                from game_headers
                group by key
                order by n desc, key
                limit 100
                """
            ).fetchall()]
        return {"db": str(db_path), "counts": counts, "archives": archives, "results": results, "header_keys": header_keys}

    @app.post("/api/query")
    def query(req: QueryRequest) -> dict[str, Any]:
        limit = max(1, min(int(req.limit), 5000))
        sql = validate_readonly_sql(req.sql)
        lowered = sql.lstrip().lower()
        executable = sql if lowered.startswith("pragma") else f"select * from ({sql}) limit {limit}"
        with connect(db_path) as conn:
            try:
                cursor = conn.execute(executable)
                rows = cursor.fetchmany(limit) if lowered.startswith("pragma") else cursor.fetchall()
            except sqlite3.Error as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = rows_to_payload(cursor, rows)
        payload["limit"] = limit
        return payload

    @app.get("/api/game/{game_id}/pgn", response_class=PlainTextResponse)
    def game_pgn(game_id: int) -> str:
        with connect(db_path) as conn:
            row = conn.execute(
                "select shard_path, byte_start, byte_end from games where game_id = ?",
                (game_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"game_id not found: {game_id}")
        shard_path = Path(row["shard_path"])
        if not shard_path.exists():
            raise HTTPException(status_code=404, detail=f"PGN shard not found: {shard_path}")
        with shard_path.open("rb") as handle:
            handle.seek(int(row["byte_start"]))
            data = handle.read(int(row["byte_end"]) - int(row["byte_start"]))
        return data.decode("utf-8", errors="replace")

    @app.get("/api/presets/{name}")
    def preset(name: str, limit: int = Query(200, ge=1, le=5000)) -> dict[str, Any]:
        presets = {
            "bse-1600-1800": "select game_id, white, black, result, white_elo, black_elo, date, source_archive from games where min_elo >= 1600 and max_elo < 1800 order by game_id",
            "dev-1800-2200": "select game_id, white, black, result, white_elo, black_elo, date, source_archive from games where min_elo >= 1800 and max_elo <= 2200 order by game_id",
            "planning-2200-plus": "select game_id, white, black, result, white_elo, black_elo, date, source_archive from games where min_elo >= 2200 order by game_id",
            "headers": "select key, count(*) as n from game_headers group by key order by n desc, key",
            "archives": "select source_archive, count(*) as games from games group by source_archive order by source_archive",
        }
        if name not in presets:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {name}")
        return query(QueryRequest(sql=presets[name], limit=limit))

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    app = make_app(args.db)
    print(f"serving Lumbras metadata store: {args.db}")
    print(f"open: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
