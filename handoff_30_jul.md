# Handoff — 30 Jul

## Current repo state

Current branch during this work:

```text
main
```

Relevant work completed today:

- scaled the BSE neighbor-index launcher from the old 100k exploratory index to a 1M index;
- moved large index/cache outputs to `/700gpart`;
- added reusable eligible-game cache support for the BSE neighbor index builder;
- fixed the 1M sampler so it does not materialize hundreds of millions of Python `(game_id, ply)` tuples;
- added a full Lumbras OTB SQLite metadata-store builder with parallel archive processing;
- created Scryer tickets for deferred final-run planning and the metadata store.

Expected local untracked/non-source paths may include:

```text
data/
tools/stockfish/
uv.lock
wandb/
```

Do not commit generated data/checkpoints/indexes unless explicitly asked.

## 1M BSE neighbor index

Top-level launcher:

```text
build_bse_neighbor_index.sh
```

It now builds a 1M BSE nearest-neighbor index from the same 1800-2200 PGN used by the old 100k index:

```text
data/processed/lumbras/lumbras_otb_both_1800_to_2200_base.pgn
```

Default output:

```text
/700gpart/chess/data/analysis/bse_neighbors_1800_2200_1m
```

Default reusable eligible-game cache:

```text
/700gpart/chess/data/processed/lumbras/bse_neighbors_1800_2200_eligible_games
```

Default sample counts:

```text
nonterminal: 900,000
terminal:    100,000
```

Default batch size:

```text
32
```

Default accepted-game cap:

```text
0  # no cap; use full available 1800-2200 PGN/cache
```

Run command:

```bash
./build_bse_neighbor_index.sh
```

Useful overrides:

```bash
BSE_NEIGHBOR_BATCH_SIZE=16 ./build_bse_neighbor_index.sh
BSE_NEIGHBOR_OUT_DIR=/path/to/index ./build_bse_neighbor_index.sh
BSE_NEIGHBOR_ELIGIBLE_GAMES_DIR=/path/to/cache ./build_bse_neighbor_index.sh
```

### Eligible-game cache behavior

`scripts/build_bse_neighbor_index.py` now accepts:

```text
--eligible-games-dir
```

If this directory contains all required files, the builder loads it instead of scanning/tokenizing the PGN again:

```text
moves.npy
offsets.npy
pgn_texts.jsonl
game_headers.jsonl
manifest.json
```

If missing/incomplete, it reads and tokenizes all accepted 1800-2200 games, writes the cache, then samples the 1M index positions.

The index `games/` path is a symlink to the reusable eligible-game cache when `--eligible-games-dir` is used, so the existing explorer can still load the index normally.

### Important sampler fix

When the user ran the 1M build, it printed:

```text
loading eligible games cache from /700gpart/chess/data/processed/lumbras/bse_neighbors_1800_2200_eligible_games
loaded 4,270,024 eligible games; total plies=336,007,688
```

Then it appeared to hang. Cause: the old `build_samples()` materialized every possible non-terminal `(game_id, ply)` tuple into Python lists before sampling. With the full cache this meant hundreds of millions of tuples.

Fix: `build_samples()` now uses vectorized per-bucket capacity counts and only materializes the requested sampled metadata rows. It should now quickly print a line like:

```text
sampled 1,000,000 positions (900,000 nonterminal, 100,000 terminal)
```

then proceed to:

```text
encoding BSE prefixes
```

## Lumbras SQLite metadata store

New files:

```text
build_lumbras_metadata_store.sh
scripts/catalog_lumbras_pgn_sqlite.py
```

Purpose: create a reusable full-corpus metadata catalog for Lumbras OTB, independent of any particular Elo extract. This avoids repeated whole-PGN/archive scans for future experiments.

One-time flow:

```text
raw .7z archives -> uncompressed seekable PGN shards -> SQLite metadata + byte offsets
```

Later extraction flow:

```text
SQL query -> game ids / shard_path / byte_start / byte_end -> seek/read selected PGNs only -> tokenize/replay/cache/build index
```

Default output:

```text
/700gpart/chess/data/catalog/lumbras_otb/
  lumbras_otb_catalog.sqlite
  pgn_shards/
    <archive-stem>.pgn
  worker_dbs/
    <archive-stem>.sqlite
```

Run command:

```bash
./build_lumbras_metadata_store.sh
```

Force rebuild:

```bash
LUMBRAS_METADATA_FORCE=1 ./build_lumbras_metadata_store.sh
```

Default worker count:

```text
12
```

Override:

```bash
LUMBRAS_METADATA_WORKERS=8 ./build_lumbras_metadata_store.sh
```

### Metadata-store multiprocessing design

The metadata builder now supports `--workers`.

Parallel mode behavior:

1. process multiple `.7z` archives concurrently;
2. each worker streams one archive, writes one uncompressed PGN shard, and writes a per-archive worker SQLite DB;
3. the parent process merges finished worker DBs into the final catalog DB;
4. final indexes are created after merge.

This avoids SQLite write contention while still using multiple CPU/archive streams.

### SQLite schema

Primary table:

```sql
games(
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
)
```

Full queryable headers table:

```sql
game_headers(
  game_id integer not null,
  key text not null,
  value text not null,
  primary key (game_id, key),
  foreign key (game_id) references games(game_id)
)
```

Initial indexes:

```sql
idx_games_elo on games(min_elo, max_elo)
idx_games_year on games(year)
idx_games_result on games(result)
idx_games_archive on games(source_archive)
idx_games_ply_count on games(ply_count_header)
idx_game_headers_key_value on game_headers(key, value)
idx_game_headers_key on game_headers(key)
```

The `.7z` archives are only streamed during catalog construction. Runtime readers should use the uncompressed `pgn_shards/*.pgn` paths and byte offsets from SQLite.

## Scryer tickets created

Project:

```text
Chess
ID: ee5d9e92-440e-4df1-8b6f-1fd4768c3526
```

### Final Run when everything is ready

```text
ID: 644e5c84-6dce-4450-8201-e73e308f7cb2
Status: unopened
Type: Research
```

Documents deferred final-run thoughts:

- current priority remains Manifolder/z2;
- validated `256d/48Q` bf16/fp16 BSE;
- BSE is compute-bound more than VRAM-bound;
- Vast candidates: 4070 Ti Super or 5070 Ti-class if CUDA/PyTorch support is stable;
- final data hygiene split idea:

```text
1600-1800   BSE training
1800-2200   z1/z2 neighbor indexes and Manifolder/dev
2200+       stronger-player manifold/planning training
held-outs   fixed eval buckets across Elo bands
```

### Metadata Store

```text
ID: 946b54b0-c4f1-40f1-9428-ac528c757970
Status: unopened
Type: Feature
```

Documents full SQLite metadata store design, schema, indexes, shard/byte-offset behavior, and downstream usage.

## Next immediate task

The user asked to check in/push everything and keep this handoff updated before building a small interface to query the SQLite metadata store.

After commit/push, next likely work:

```text
small SQLite query interface for Lumbras metadata store
```

Likely useful features:

- show DB summary counts;
- run canned queries for Elo bands (`1600-1800`, `1800-2200`, `2200+`);
- run arbitrary read-only SQL;
- optionally fetch/display PGN text for selected `game_id` using `shard_path` + byte offsets.
