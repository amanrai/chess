import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "preprocess_board_state_verifier.py"
SPEC = importlib.util.spec_from_file_location("preprocess_board_state_verifier", SCRIPT)
board_preprocess = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(board_preprocess)


class Piece:
    def __init__(self, piece_type, color):
        self.piece_type = piece_type
        self.color = color


class Board:
    def __init__(self, pieces):
        self.pieces = pieces

    def piece_at(self, square):
        return self.pieces.get(square)


class BoardStatePreprocessTests(unittest.TestCase):
    def test_pack_board_uses_a1_to_h8_and_two_nibbles_per_byte(self):
        # a1 white pawn -> low nibble; b1 black king -> high nibble.
        packed = board_preprocess.pack_board_after(Board({0: Piece(1, True), 1: Piece(6, False)}))
        self.assertEqual(packed.shape, (32,))
        self.assertEqual(int(packed[0]), 0xC1)
        self.assertTrue(np.all(packed[1:] == 0))

    def test_bucket_plan_only_selects_games_reaching_the_prefix(self):
        lengths = np.asarray([21, 25, 35], dtype=np.int64)
        plan, records = board_preprocess.bucket_plan(
            lengths, samples=101, bucket_plies=10, max_prefix_plies=35,
            allocation_alpha=1.15, min_samples_per_bucket=0, seed=7,
        )
        self.assertEqual(plan.shape, (101, 2))
        self.assertEqual(sum(record["samples"] for record in records), 101)
        self.assertEqual([record["eligible_games"] for record in records], [3, 3, 3, 1])
        for game_i, prefix_plies in plan:
            self.assertLess(int(game_i), len(lengths))
            self.assertGreaterEqual(int(prefix_plies), 1)
            self.assertLessEqual(int(prefix_plies), int(lengths[game_i]))

    def test_bucket_plan_is_seed_deterministic(self):
        kwargs = dict(samples=50, bucket_plies=10, max_prefix_plies=30, allocation_alpha=1.0,
                      min_samples_per_bucket=0, seed=123)
        first, _ = board_preprocess.bucket_plan(np.asarray([21, 30]), **kwargs)
        second, _ = board_preprocess.bucket_plan(np.asarray([21, 30]), **kwargs)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
