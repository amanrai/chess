import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "histogram_game_plies.py"
SPEC = importlib.util.spec_from_file_location("histogram_game_plies", SCRIPT)
histogram = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(histogram)


class GamePlyHistogramTests(unittest.TestCase):
    def test_five_move_buckets_cover_zero_and_odd_ply_lengths(self):
        records = histogram.build_histogram(np.asarray([0, 1, 10, 11, 20, 21]), 5)

        self.assertEqual(
            records,
            [
                {
                    "bucket_index": 0,
                    "moves_lo": 0,
                    "moves_hi": 5,
                    "plies_lo": 0,
                    "plies_hi": 10,
                    "games": 3,
                },
                {
                    "bucket_index": 1,
                    "moves_lo": 6,
                    "moves_hi": 10,
                    "plies_lo": 11,
                    "plies_hi": 20,
                    "games": 2,
                },
                {
                    "bucket_index": 2,
                    "moves_lo": 11,
                    "moves_hi": 15,
                    "plies_lo": 21,
                    "plies_hi": 30,
                    "games": 1,
                },
            ],
        )

    def test_summary_handles_empty_lengths(self):
        self.assertEqual(
            histogram.length_summary(np.asarray([], dtype=np.int64)),
            {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
