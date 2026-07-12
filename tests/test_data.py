import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from chessgm.data import InverseTransitionDataset, OutcomeConditionedTransitionDataset


class InverseTransitionDatasetTests(unittest.TestCase):
    def test_samples_immediate_successor_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moves = np.asarray(
                [
                    [1] * 8,
                    [2] * 8,
                    [3] * 8,
                ],
                dtype=np.uint16,
            )
            np.save(root / "moves.npy", moves)
            np.save(root / "offsets.npy", np.asarray([0, 3], dtype=np.int64))
            np.save(root / "results.npy", np.asarray([0], dtype=np.int64))
            dataset = InverseTransitionDataset(root, context_plies=4, examples_per_epoch=1)
            before, after, target = dataset[0]

        self.assertEqual(before.shape, (4, 8))
        self.assertEqual(after.shape, (4, 8))
        self.assertEqual(target.shape, (8,))
        self.assertTrue(torch.equal(after[-1], target))
        self.assertFalse(torch.equal(before[-1], target))
        self.assertTrue(torch.equal(after[-2], before[-1]))

    def test_conditions_target_side_on_matching_game_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            white_game = np.asarray([[100 + ply] * 8 for ply in range(6)], dtype=np.uint16)
            black_game = np.asarray([[200 + ply] * 8 for ply in range(6)], dtype=np.uint16)
            np.save(root / "moves.npy", np.concatenate([white_game, black_game]))
            np.save(root / "offsets.npy", np.asarray([0, 6, 12], dtype=np.int64))
            np.save(root / "results.npy", np.asarray([0, 1], dtype=np.int64))
            dataset = OutcomeConditionedTransitionDataset(
                root,
                context_plies=4,
                examples_per_epoch=20,
                max_transition_plies=4,
            )
            targets = [int(dataset[index][2][0]) for index in range(len(dataset))]

        self.assertEqual(dataset.target_plies, [1, 2, 3, 4])
        self.assertTrue(any(token < 200 for token in targets))
        self.assertTrue(any(token >= 200 for token in targets))
        for token in targets:
            if token < 200:
                self.assertEqual((token - 100) % 2, 0)  # white to move, white-win game
            else:
                self.assertEqual((token - 200) % 2, 1)  # black to move, black-win game


if __name__ == "__main__":
    unittest.main()
