import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from chessgm.data import InverseTransitionDataset


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


if __name__ == "__main__":
    unittest.main()
