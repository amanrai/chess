#!/usr/bin/env python3
"""Train the Q-Former style verifier encoder."""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from chessgm.data import VerifierGameStoreDataset
from chessgm.network_q import QVerifierTransformer
from chessgm.tokenizer import VOCAB


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().detach().cpu())


def class_precision_counts(logits: torch.Tensor, y: torch.Tensor) -> tuple[list[int], list[int]]:
    pred = logits.argmax(dim=-1)
    tp, predicted = [], []
    for cls in range(3):
        pred_mask = pred == cls
        tp.append(int((pred_mask & (y == cls)).sum().detach().cpu()))
        predicted.append(int(pred_mask.sum().detach().cpu()))
    return tp, predicted


def rolling_precisions(window: deque[tuple[list[int], list[int]]]) -> dict[str, str]:
    names = ["white", "black", "draw"]
    tp = [0, 0, 0]
    predicted = [0, 0, 0]
    for batch_tp, batch_predicted in window:
        for i in range(3):
            tp[i] += batch_tp[i]
            predicted[i] += batch_predicted[i]
    return {
        f"p_{name}": f"{(tp[i] / predicted[i] if predicted[i] else 0.0):.3f} [{tp[i]}/{predicted[i]}]"
        for i, name in enumerate(names)
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    batch: int,
    global_batch: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "args": vars(args),
            "vocab": VOCAB,
            "epoch": epoch,
            "batch": batch,
            "global_batch": global_batch,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "lumbras" / "verifier")
    parser.add_argument("--context-moves", type=int, default=128)
    parser.add_argument("--sample-mode", choices=["prefix", "full"], default="prefix")
    parser.add_argument(
        "--bucket-mode",
        choices=["fraction", "absolute"],
        default="fraction",
        help="Prefix sampling buckets used when --sample-mode prefix",
    )
    parser.add_argument(
        "--min-game-moves",
        type=int,
        default=None,
        help="Exclude games with fewer than this many move packets/plies before sampling",
    )
    parser.add_argument(
        "--max-game-moves",
        type=int,
        default=None,
        help="Exclude games with more than this many move packets/plies before sampling",
    )
    parser.add_argument(
        "--prefix-fraction",
        type=float,
        default=None,
        help="Use exactly this fraction of each game as the prefix, e.g. 0.5 for half-game",
    )
    parser.add_argument(
        "--prefix-fraction-min",
        type=float,
        default=None,
        help="Minimum random prefix fraction; used with --prefix-fraction-max",
    )
    parser.add_argument(
        "--prefix-fraction-max",
        type=float,
        default=None,
        help="Maximum random prefix fraction; used with --prefix-fraction-min",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--history-layers", type=int, default=4)
    parser.add_argument("--q-layers", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--examples-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "q_verifier")
    parser.add_argument(
        "--snapshot-every-batches",
        type=int,
        default=5000,
        help="Save an in-epoch snapshot every N batches; 0 disables",
    )
    parser.add_argument("--log-window", type=int, default=1000)
    args = parser.parse_args()

    dataset = VerifierGameStoreDataset(
        args.data_dir,
        context_moves=args.context_moves,
        examples_per_epoch=args.examples_per_epoch,
        sample_mode=args.sample_mode,
        bucket_mode=args.bucket_mode,
        min_game_moves=args.min_game_moves,
        max_game_moves=args.max_game_moves,
        prefix_fraction=args.prefix_fraction,
        prefix_fraction_min=args.prefix_fraction_min,
        prefix_fraction_max=args.prefix_fraction_max,
    )
    print(
        "dataset: "
        f"games={len(dataset.game_indices):,} "
        f"examples_per_epoch={len(dataset):,} "
        f"sample_mode={args.sample_mode} "
        f"bucket_mode={args.bucket_mode} "
        f"prefix_fraction={args.prefix_fraction} "
        f"prefix_fraction_range=({args.prefix_fraction_min}, {args.prefix_fraction_max}) "
        f"context_moves={args.context_moves} "
        f"min_game_moves={args.min_game_moves} "
        f"max_game_moves={args.max_game_moves}"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = QVerifierTransformer(
        vocab_size=len(VOCAB),
        move_expr=8,
        model_dim=args.model_dim,
        heads=args.heads,
        history_layers=args.history_layers,
        q_layers=args.q_layers,
        num_queries=args.num_queries,
        dropout=args.dropout,
        pad_id=0,
    ).to(args.device)

    total_params, trainable_params = count_parameters(model)
    print(f"model params: total={total_params / 1e6:.3f}M trainable={trainable_params / 1e6:.3f}M")

    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps must be >= 1, got {args.grad_accum_steps}")
    if args.snapshot_every_batches < 0:
        raise ValueError(f"--snapshot-every-batches must be >= 0, got {args.snapshot_every_batches}")
    print(
        f"optimization: batch_size={args.batch_size} "
        f"grad_accum_steps={args.grad_accum_steps} "
        f"effective_batch_size={args.batch_size * args.grad_accum_steps}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_batch = 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_window: deque[float] = deque(maxlen=args.log_window)
        acc_window: deque[float] = deque(maxlen=args.log_window)
        precision_window: deque[tuple[list[int], list[int]]] = deque(maxlen=args.log_window)
        pbar = tqdm(loader, desc=f"q-verifier epoch {epoch + 1}/{args.epochs}", unit="batch")
        for step, (x, y) in enumerate(pbar, start=1):
            global_batch += 1
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            (loss / args.grad_accum_steps).backward()

            if step % args.grad_accum_steps == 0 or step == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            loss_value = float(loss.detach().cpu())
            acc_value = accuracy(logits, y)
            loss_window.append(loss_value)
            acc_window.append(acc_value)
            precision_window.append(class_precision_counts(logits, y))
            rolling_loss = sum(loss_window) / len(loss_window)
            pbar.set_postfix(loss=rolling_loss)
            if step % 100 == 0 or step == len(loader):
                precisions = rolling_precisions(precision_window)
                metrics = {
                    "loss": f"{rolling_loss:.4f}",
                    "acc": f"{sum(acc_window) / len(acc_window):.4f}",
                    **precisions,
                }
                metrics_text = " ".join(f"{k}={v}" for k, v in metrics.items())
                cyan = "\033[36m"
                reset = "\033[0m"
                pbar.write(f"{cyan}epoch={epoch + 1} batch={step}/{len(loader)} {metrics_text}{reset}")
                precision_window.clear()

            if args.snapshot_every_batches and step % args.snapshot_every_batches == 0:
                snapshot_path = (
                    args.checkpoint_dir / f"q_verifier_epoch_{epoch + 1:03d}_batch_{step:06d}.pt"
                )
                save_checkpoint(
                    snapshot_path,
                    model=model,
                    opt=opt,
                    args=args,
                    epoch=epoch + 1,
                    batch=step,
                    global_batch=global_batch,
                )
                pbar.write(f"saved snapshot: {snapshot_path}")

        ckpt_path = args.checkpoint_dir / f"q_verifier_epoch_{epoch + 1}.pt"
        save_checkpoint(
            ckpt_path,
            model=model,
            opt=opt,
            args=args,
            epoch=epoch + 1,
            batch=len(loader),
            global_batch=global_batch,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
