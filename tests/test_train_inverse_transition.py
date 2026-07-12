import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_inverse_transition.py"


class TrainInverseTransitionTests(unittest.TestCase):
    def test_writes_snapshot_and_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            checkpoint_dir = root / "checkpoints"
            data_dir.mkdir()
            moves = np.asarray([[token] * 8 for token in range(1, 6)], dtype=np.uint16)
            np.save(data_dir / "moves.npy", moves)
            np.save(data_dir / "offsets.npy", np.asarray([0, 5], dtype=np.int64))
            np.save(data_dir / "results.npy", np.asarray([0], dtype=np.int64))
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-dir",
                    str(data_dir),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--examples-per-epoch",
                    "2",
                    "--batch-size",
                    "2",
                    "--epochs",
                    "1",
                    "--context-plies",
                    "3",
                    "--model-dim",
                    "16",
                    "--heads",
                    "4",
                    "--history-layers",
                    "1",
                    "--q-layers",
                    "1",
                    "--num-queries",
                    "4",
                    "--transition-layers",
                    "1",
                    "--snapshot-every-batches",
                    "1",
                    "--device",
                    "cpu",
                ],
                check=True,
                cwd=ROOT,
            )
            snapshot = checkpoint_dir / "inverse_transition_epoch_001_batch_000001.pt"
            epoch = checkpoint_dir / "inverse_transition_epoch_1.pt"
            self.assertTrue(snapshot.exists())
            self.assertTrue(epoch.exists())
            checkpoint = torch.load(epoch, map_location="cpu", weights_only=False)

        required = {"model", "optimizer", "args", "epoch", "batch", "global_batch"}
        self.assertEqual(set(checkpoint).intersection(required), required)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["batch"], 1)
        self.assertEqual(checkpoint["global_batch"], 1)


if __name__ == "__main__":
    unittest.main()
