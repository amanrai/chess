"""Datasets for preprocessed chess model arrays."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PrefixBucket:
    name: str
    lo: float
    hi: float
    weight: float = 1.0


FRACTION_PREFIX_BUCKETS = [
    PrefixBucket("first_20pct", 0.00, 0.20, 1.0),
    PrefixBucket("middle_30pct", 0.20, 0.50, 1.0),
    PrefixBucket("late_30pct", 0.50, 0.80, 1.0),
    PrefixBucket("final_20pct", 0.80, 1.00, 1.0),
]

ABSOLUTE_PREFIX_BUCKETS = [
    PrefixBucket("opening_1_16", 1, 16, 1.0),
    PrefixBucket("early_mid_17_40", 17, 40, 1.0),
    PrefixBucket("midgame_41_80", 41, 80, 1.0),
    PrefixBucket("late_81_plus", 81, 10**9, 1.0),
]


def bucket_to_prefix_range(
    num_moves: int,
    bucket: PrefixBucket,
    mode: str,
) -> tuple[int, int] | None:
    if num_moves <= 0:
        return None
    if mode == "absolute":
        lo = int(bucket.lo)
        hi = int(min(bucket.hi, num_moves))
    elif mode == "fraction":
        lo = max(1, int(num_moves * bucket.lo) + 1)
        hi = max(lo, int(num_moves * bucket.hi))
        hi = min(hi, num_moves)
    else:
        raise ValueError(f"unknown bucket mode: {mode}")
    if lo > num_moves or hi < lo:
        return None
    return lo, hi


def weighted_choice(items, weights, rng: random.Random):
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for item, weight in zip(items, weights):
        upto += weight
        if upto >= r:
            return item
    return items[-1]


class VerifierGameStoreDataset(Dataset):
    """Dynamically sample verifier prefixes from game-store arrays.

    Expects files created by scripts/preprocess_verifier_dataset.py:
      moves.npy, offsets.npy, results.npy
    """

    def __init__(
        self,
        root: str | Path,
        context_moves: int = 128,
        pad_id: int = 0,
        bucket_mode: str = "fraction",
        buckets: list[PrefixBucket] | None = None,
        examples_per_epoch: int | None = None,
        seed: int = 0,
        sample_mode: str = "prefix",
        min_game_moves: int | None = None,
        max_game_moves: int | None = None,
        prefix_fraction: float | None = None,
        prefix_fraction_min: float | None = None,
        prefix_fraction_max: float | None = None,
    ):
        self.root = Path(root)
        self.moves = np.load(self.root / "moves.npy", mmap_mode="r")
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self.results = np.load(self.root / "results.npy", mmap_mode="r")
        self.context_moves = context_moves
        self.pad_id = pad_id
        self.bucket_mode = bucket_mode
        self.buckets = buckets or (FRACTION_PREFIX_BUCKETS if bucket_mode == "fraction" else ABSOLUTE_PREFIX_BUCKETS)
        self.seed = seed
        self.move_expr = int(self.moves.shape[1])
        self.sample_mode = sample_mode
        self.min_game_moves = min_game_moves
        self.max_game_moves = max_game_moves
        self.prefix_fraction = prefix_fraction
        self.prefix_fraction_min = prefix_fraction_min
        self.prefix_fraction_max = prefix_fraction_max

        if sample_mode not in {"prefix", "full"}:
            raise ValueError(f"unknown sample_mode={sample_mode!r}; expected 'prefix' or 'full'")
        if prefix_fraction is not None and (
            prefix_fraction_min is not None or prefix_fraction_max is not None
        ):
            raise ValueError("use either prefix_fraction or prefix_fraction_min/max, not both")
        for name, value in {
            "prefix_fraction": prefix_fraction,
            "prefix_fraction_min": prefix_fraction_min,
            "prefix_fraction_max": prefix_fraction_max,
        }.items():
            if value is not None and not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if (
            prefix_fraction_min is not None
            and prefix_fraction_max is not None
            and prefix_fraction_min > prefix_fraction_max
        ):
            raise ValueError("prefix_fraction_min must be <= prefix_fraction_max")
        game_lengths = np.diff(self.offsets)
        valid_mask = np.ones(len(self.results), dtype=bool)
        if min_game_moves is not None:
            valid_mask &= game_lengths >= min_game_moves
        if max_game_moves is not None:
            valid_mask &= game_lengths <= max_game_moves
        self.game_indices = np.flatnonzero(valid_mask).astype(np.int64)
        if len(self.game_indices) == 0:
            raise ValueError(
                "no verifier games left after filtering "
                f"min_game_moves={min_game_moves} max_game_moves={max_game_moves}"
            )
        self.examples_per_epoch = examples_per_epoch or len(self.game_indices)

    def __len__(self) -> int:
        return self.examples_per_epoch

    def sample_prefix_length(self, num_moves: int, rng: random.Random) -> int:
        if self.prefix_fraction is not None:
            return max(1, min(num_moves, math.ceil(num_moves * self.prefix_fraction)))
        if self.prefix_fraction_min is not None or self.prefix_fraction_max is not None:
            lo = self.prefix_fraction_min if self.prefix_fraction_min is not None else 0.0
            hi = self.prefix_fraction_max if self.prefix_fraction_max is not None else 1.0
            fraction = rng.uniform(lo, hi)
            return max(1, min(num_moves, math.ceil(num_moves * fraction)))

        valid = [b for b in self.buckets if bucket_to_prefix_range(num_moves, b, self.bucket_mode)]
        if not valid:
            return rng.randint(1, num_moves)
        bucket = weighted_choice(valid, [b.weight for b in valid], rng)
        lo, hi = bucket_to_prefix_range(num_moves, bucket, self.bucket_mode)  # type: ignore[misc]
        return rng.randint(lo, hi)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Deterministic per index/epoch-ish sampling. Later trainer can vary seed per epoch.
        rng = random.Random(self.seed + idx)
        if self.sample_mode == "full" and self.examples_per_epoch == len(self.game_indices):
            game_i = int(self.game_indices[idx % len(self.game_indices)])
        else:
            game_i = int(self.game_indices[rng.randrange(len(self.game_indices))])
        start = int(self.offsets[game_i])
        end = int(self.offsets[game_i + 1])
        num_moves = end - start

        if self.sample_mode == "full":
            prefix = self.moves[start:end]
        else:
            prefix_len = self.sample_prefix_length(num_moves, rng)
            prefix = self.moves[start : start + prefix_len]

        if len(prefix) >= self.context_moves:
            x = prefix[-self.context_moves :]
        else:
            pad_rows = np.full(
                (self.context_moves - len(prefix), self.move_expr),
                self.pad_id,
                dtype=np.uint16,
            )
            x = np.concatenate([pad_rows, prefix], axis=0)

        y = int(self.results[game_i])
        return torch.from_numpy(x.astype(np.int64)), torch.tensor(y, dtype=torch.long)
