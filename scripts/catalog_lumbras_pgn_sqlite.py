#!/usr/bin/env python3
"""Build a reusable SQLite metadata catalog for Lumbras PGN archives.

The catalog is intentionally independent of a particular Elo/indexing experiment.
It streams the compressed .7z archives once, writes seekable uncompressed PGN
shards, and records byte offsets plus parsed headers in SQLite. Later pipelines can
use SQL to pick candidate games and read only those PGN byte ranges.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_lumbras_2200_splits import archive_stream

HEADER_RE = re.compile(r'^\[([^\s]+) "(.*)"\]$')
DEFAULT_RAW_DIR = Path("data/raw/lumbras/otb")
DEFAULT_OUT_DIR = Path("/700gpart/chess/data/catalog/lumbras_otb")


def parse_int_header(headers: dict[str, str], key: str) -> int | None:
    value = headers.get(key)
    if not value or value in {"?", "-"}:
        return None
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def parse_year(headers: dict[str, str]) -> int | None:
    date = headers.get("Date") or headers.get("UTCDate")
    if not date:
        return None
    match = re.match(r"^(\d{4})", date)
    return int(match.group(1)) if match else None


def parse_headers(lines: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in lines:
        if not line.startswith("["):
            break
        match = HEADER_RE.match(line.rstrip("\n"))
        if match:
            headers[match.group(1)] = match.group(2)
    return headers


def iter_game_lines(path: Path) -> Iterable[list[str]]:
    buf: list[str] = []
    for line in archive_stream(path):
        if line.startswith("[Event ") and buf:
            yield buf
            buf = []
        buf.append(line)
    if buf:
        yield buf


def init_db(conn: sqlite3.Connection, *, create_indexes: bool = True) -> None:
    conn.executescript(
        """
        pragma journal_mode = wal;
        pragma synchronous = normal;

        create table if not exists games (
            game_id integer primary key,
            source_archive text not null,
            shard_path text not null,
            byte_start integer not null,
            byte_end integer not null,
            byte_len integer not null,
            event text,
            site text,
            date text,
            year integer,
            round text,
            white text,
            black text,
            result text,
            white_elo integer,
            black_elo integer,
            min_elo integer,
            max_elo integer,
            eco text,
            opening text,
            ply_count_header integer,
            headers_json text not null
        );

        create table if not exists game_headers (
            game_id integer not null,
            key text not null,
            value text not null,
            primary key (game_id, key),
            foreign key (game_id) references games(game_id)
        );

        """
    )
    if create_indexes:
        create_db_indexes(conn)


def create_db_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create index if not exists idx_games_elo on games(min_elo, max_elo);
        create index if not exists idx_games_year on games(year);
        create index if not exists idx_games_result on games(result);
        create index if not exists idx_games_archive on games(source_archive);
        create index if not exists idx_games_ply_count on games(ply_count_header);
        create index if not exists idx_game_headers_key_value on game_headers(key, value);
        create index if not exists idx_game_headers_key on game_headers(key);
        """
    )


def already_cataloged(conn: sqlite3.Connection, archive_name: str) -> bool:
    row = conn.execute("select 1 from games where source_archive = ? limit 1", (archive_name,)).fetchone()
    return row is not None


def catalog_archive(conn: sqlite3.Connection, archive: Path, shards_dir: Path, commit_every: int) -> int:
    archive_name = archive.name
    shard_path = shards_dir / f"{archive.stem}.pgn"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    game_rows: list[tuple] = []
    header_rows: list[tuple] = []
    next_game_id = int(conn.execute("select coalesce(max(game_id), -1) + 1 from games").fetchone()[0])

    print(f"Scanning {archive} -> {shard_path}")
    with shard_path.open("wb") as shard:
        for lines in iter_game_lines(archive):
            headers = parse_headers(lines)
            text = "".join(lines)
            if not text.endswith("\n"):
                text += "\n"
            text += "\n"
            data = text.encode("utf-8", errors="replace")
            byte_start = shard.tell()
            shard.write(data)
            byte_end = shard.tell()

            white_elo = parse_int_header(headers, "WhiteElo")
            black_elo = parse_int_header(headers, "BlackElo")
            elos = [e for e in (white_elo, black_elo) if e is not None]
            min_elo = min(elos) if len(elos) == 2 else None
            max_elo = max(elos) if len(elos) == 2 else None
            ply_count = parse_int_header(headers, "PlyCount")
            game_id = next_game_id
            next_game_id += 1
            game_rows.append(
                (
                    game_id,
                    archive_name,
                    str(shard_path),
                    byte_start,
                    byte_end,
                    byte_end - byte_start,
                    headers.get("Event"),
                    headers.get("Site"),
                    headers.get("Date"),
                    parse_year(headers),
                    headers.get("Round"),
                    headers.get("White"),
                    headers.get("Black"),
                    headers.get("Result"),
                    white_elo,
                    black_elo,
                    min_elo,
                    max_elo,
                    headers.get("ECO"),
                    headers.get("Opening"),
                    ply_count,
                    json.dumps(headers, ensure_ascii=False),
                )
            )
            header_rows.extend((game_id, key, value) for key, value in headers.items())
            count += 1
            if len(game_rows) >= commit_every:
                insert_rows(conn, game_rows, header_rows)
                game_rows.clear()
                header_rows.clear()
                print(f"  {archive_name}: {count:,} games", end="\r", flush=True)
    if game_rows:
        insert_rows(conn, game_rows, header_rows)
    print(f"  {archive_name}: {count:,} games")
    return count


def insert_rows(conn: sqlite3.Connection, game_rows: list[tuple], header_rows: list[tuple]) -> None:
    conn.executemany(
        """
        insert into games (
            game_id, source_archive, shard_path, byte_start, byte_end, byte_len,
            event, site, date, year, round, white, black, result,
            white_elo, black_elo, min_elo, max_elo, eco, opening,
            ply_count_header, headers_json
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        game_rows,
    )
    conn.executemany(
        "insert into game_headers (game_id, key, value) values (?, ?, ?)",
        header_rows,
    )
    conn.commit()


def catalog_archive_worker(payload: tuple[str, str, str, int, bool]) -> tuple[str, int, str]:
    archive_s, shards_dir_s, worker_db_s, commit_every, force = payload
    archive = Path(archive_s)
    worker_db = Path(worker_db_s)
    done_path = worker_db.with_suffix(worker_db.suffix + ".done")
    if done_path.exists() and worker_db.exists() and not force:
        conn = sqlite3.connect(worker_db)
        count = int(conn.execute("select count(*) from games").fetchone()[0])
        conn.close()
        print(f"Skipping already built worker DB: {archive.name} ({count:,} games)")
        return archive.name, count, str(worker_db)

    worker_db.parent.mkdir(parents=True, exist_ok=True)
    worker_db.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    conn = sqlite3.connect(worker_db)
    init_db(conn, create_indexes=False)
    count = catalog_archive(conn, archive, Path(shards_dir_s), commit_every)
    conn.close()
    done_path.write_text(json.dumps({"archive": archive.name, "games": count}) + "\n", encoding="utf-8")
    return archive.name, count, str(worker_db)


def merge_worker_db(conn: sqlite3.Connection, worker_db: Path) -> int:
    worker_conn = sqlite3.connect(worker_db)
    archive_name = worker_conn.execute("select source_archive from games limit 1").fetchone()
    game_count = int(worker_conn.execute("select count(*) from games").fetchone()[0])
    worker_conn.close()
    if archive_name is None:
        return 0
    archive = str(archive_name[0])
    if already_cataloged(conn, archive):
        print(f"Skipping already merged archive: {archive}")
        return 0
    offset = int(conn.execute("select coalesce(max(game_id), -1) + 1 from games").fetchone()[0])
    conn.execute("attach database ? as worker", (str(worker_db),))
    try:
        conn.execute(
            """
            insert into games (
                game_id, source_archive, shard_path, byte_start, byte_end, byte_len,
                event, site, date, year, round, white, black, result,
                white_elo, black_elo, min_elo, max_elo, eco, opening,
                ply_count_header, headers_json
            )
            select
                game_id + ?, source_archive, shard_path, byte_start, byte_end, byte_len,
                event, site, date, year, round, white, black, result,
                white_elo, black_elo, min_elo, max_elo, eco, opening,
                ply_count_header, headers_json
            from worker.games
            """,
            (offset,),
        )
        conn.execute(
            """
            insert into game_headers (game_id, key, value)
            select game_id + ?, key, value from worker.game_headers
            """,
            (offset,),
        )
        conn.commit()
    finally:
        conn.execute("detach database worker")
    print(f"Merged {archive}: {game_count:,} games")
    return game_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--commit-every", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1, help="Archives to process concurrently")
    parser.add_argument("--force", action="store_true", help="Rebuild DB and shards from scratch")
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()

    archives = args.archives or sorted(args.raw_dir.glob("*.7z"))
    if not archives:
        raise SystemExit(f"No .7z archives found in {args.raw_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.db or (args.out_dir / "lumbras_otb_catalog.sqlite")
    shards_dir = args.out_dir / "pgn_shards"
    worker_db_dir = args.out_dir / "worker_dbs"

    if args.force:
        db_path.unlink(missing_ok=True)
        if shards_dir.exists():
            for old in shards_dir.glob("*.pgn"):
                old.unlink()
        if worker_db_dir.exists():
            for old in worker_db_dir.glob("*.sqlite*"):
                old.unlink()

    conn = sqlite3.connect(db_path)
    init_db(conn, create_indexes=False)

    counts: Counter[str] = Counter()
    if args.workers <= 1:
        for archive in archives:
            if already_cataloged(conn, archive.name) and not args.force:
                print(f"Skipping already cataloged archive: {archive.name}")
                continue
            counts[archive.name] = catalog_archive(conn, archive, shards_dir, args.commit_every)
    else:
        worker_db_dir.mkdir(parents=True, exist_ok=True)
        todo = [archive for archive in archives if args.force or not already_cataloged(conn, archive.name)]
        for archive in archives:
            if archive not in todo:
                print(f"Skipping already cataloged archive: {archive.name}")
        payloads = [
            (
                str(archive),
                str(shards_dir),
                str(worker_db_dir / f"{archive.stem}.sqlite"),
                args.commit_every,
                args.force,
            )
            for archive in todo
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(catalog_archive_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                archive_name, count, worker_db_s = future.result()
                counts[archive_name] = count
                merge_worker_db(conn, Path(worker_db_s))

    print("creating SQLite indexes")
    create_db_indexes(conn)

    total = conn.execute("select count(*) from games").fetchone()[0]
    both_1800_2200 = conn.execute(
        "select count(*) from games where min_elo >= 1800 and max_elo <= 2200"
    ).fetchone()[0]
    both_2200_plus = conn.execute("select count(*) from games where min_elo >= 2200").fetchone()[0]

    print("\nCatalog complete")
    print(f"  db: {db_path}")
    print(f"  shards: {shards_dir}")
    print(f"  total games: {total:,}")
    print(f"  both 1800-2200: {both_1800_2200:,}")
    print(f"  both 2200+: {both_2200_plus:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
