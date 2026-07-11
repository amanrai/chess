#!/usr/bin/env python3
"""Train the verifier result classifier."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessgm.trainer import VerifierTrainConfig, train_verifier  # noqa: E402


def read_dotenv_key(path: Path = ROOT / ".env", key: str = "wandb_key") -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'") or None
    return None


def maybe_login_wandb() -> None:
    api_key = os.environ.get("WANDB_API_KEY") or read_dotenv_key()
    if not api_key:
        return
    import wandb

    wandb.login(key=api_key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "lumbras" / "verifier")
    parser.add_argument("--context-plies", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--examples-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "verifier")
    parser.add_argument("--log-window", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="chess-gm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-log-every", type=int, default=10)
    args = parser.parse_args()

    config = VerifierTrainConfig(
        data_dir=args.data_dir,
        context_plies=args.context_plies,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        model_dim=args.model_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        examples_per_epoch=args.examples_per_epoch,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        log_window=args.log_window,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_log_every=args.wandb_log_every,
    )
    if args.device is not None:
        config.device = args.device
    if args.wandb:
        maybe_login_wandb()

    train_verifier(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
