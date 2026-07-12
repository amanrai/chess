#!/usr/bin/env python3
"""Train a Q-Former inverse-transition decoder on immediate chess plies.

The shared encoder produces full Q-Former state banks for histories before and
after a target ply. The decoder asks which prior-state features (K/V) explain
the successor-state queries (Q), then predicts the complete target ply packet.
"""
from __future__ import annotations

import argparse
import os
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

from chessgm.data import OutcomeConditionedTransitionDataset
from chessgm.network_q import QInverseTransitionDecoder
from chessgm.tokenizer import TOKEN_TO_ID, VOCAB


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


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


def init_wandb(args: argparse.Namespace):
    if not args.wandb:
        return None
    api_key = os.environ.get("WANDB_API_KEY") or read_dotenv_key()
    import wandb

    if api_key:
        wandb.login(key=api_key)
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    batch: int,
    global_batch: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "processed" / "lumbras" / "verifier"
    )
    parser.add_argument("--context-plies", type=int, default=125)
    parser.add_argument("--min-game-plies", type=int, default=None)
    parser.add_argument("--max-game-plies", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--history-layers", type=int, default=4)
    parser.add_argument("--q-layers", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=16)
    parser.add_argument("--transition-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--examples-per-epoch",
        type=int,
        default=20_000_000,
        help="Outcome-conditioned transitions sampled with replacement per epoch",
    )
    parser.add_argument(
        "--max-transition-plies",
        type=int,
        default=200,
        help="Uniformly sample target plies from 1 through this limit when available",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--pretrained-encoder-checkpoint",
        type=Path,
        default=None,
        help="Optional Q-probe/Q-verifier checkpoint; only encoder.* weights are loaded",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the shared Q-Former encoder after optional checkpoint loading",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "inverse_transition"
    )
    parser.add_argument(
        "--snapshot-every-batches",
        type=int,
        default=5000,
        help="Save an in-epoch snapshot every N batches; 0 disables",
    )
    parser.add_argument("--log-window", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="chess-gm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-log-every", type=int, default=100)
    args = parser.parse_args()

    if args.context_plies < 1:
        raise ValueError("--context-plies must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.transition_layers < 0:
        raise ValueError("--transition-layers must be >= 0")
    if args.max_transition_plies < 1:
        raise ValueError("--max-transition-plies must be >= 1")
    if args.snapshot_every_batches < 0:
        raise ValueError("--snapshot-every-batches must be >= 0")
    if args.freeze_encoder and args.pretrained_encoder_checkpoint is None:
        raise ValueError("--freeze-encoder requires --pretrained-encoder-checkpoint")

    dataset = OutcomeConditionedTransitionDataset(
        args.data_dir,
        context_plies=args.context_plies,
        min_game_plies=args.min_game_plies,
        max_game_plies=args.max_game_plies,
        max_transition_plies=args.max_transition_plies,
        examples_per_epoch=args.examples_per_epoch,
    )
    white_games = len(dataset.game_indices_by_result[dataset.WHITE_WIN])
    black_games = len(dataset.game_indices_by_result[dataset.BLACK_WIN])
    print(
        "dataset: "
        f"games={len(dataset.game_indices):,} white_win_games={white_games:,} "
        f"black_win_games={black_games:,} examples_per_epoch={len(dataset):,} "
        f"context_plies={args.context_plies} transition_plies="
        f"{min(dataset.target_plies)}..{max(dataset.target_plies)} "
        f"min_game_plies={args.min_game_plies} max_game_plies={args.max_game_plies}"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = QInverseTransitionDecoder(
        vocab_size=len(VOCAB),
        ply_expr=8,
        model_dim=args.model_dim,
        heads=args.heads,
        history_layers=args.history_layers,
        q_layers=args.q_layers,
        num_queries=args.num_queries,
        transition_layers=args.transition_layers,
        dropout=args.dropout,
        pad_id=TOKEN_TO_ID["<PAD>"],
        pretrained_encoder_checkpoint=args.pretrained_encoder_checkpoint,
        freeze_encoder=args.freeze_encoder,
    ).to(args.device)
    total_params, trainable_params = count_parameters(model)
    print(
        f"model params: total={total_params / 1e6:.3f}M trainable={trainable_params / 1e6:.3f}M "
        f"loaded_encoder_tensors={model.loaded_encoder_tensors}"
    )
    print(
        f"optimization: batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps} "
        f"effective_batch_size={args.batch_size * args.grad_accum_steps}"
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args)
    if wandb_run is not None:
        wandb_run.log(
            {"params/total_m": total_params / 1e6, "params/trainable_m": trainable_params / 1e6},
            step=0,
        )

    global_batch = 0
    pad_id = TOKEN_TO_ID["<PAD>"]
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_window: deque[float] = deque(maxlen=args.log_window)
        token_correct_window: deque[int] = deque(maxlen=args.log_window)
        token_count_window: deque[int] = deque(maxlen=args.log_window)
        nonpad_correct_window: deque[int] = deque(maxlen=args.log_window)
        nonpad_count_window: deque[int] = deque(maxlen=args.log_window)
        packet_correct_window: deque[int] = deque(maxlen=args.log_window)
        packet_count_window: deque[int] = deque(maxlen=args.log_window)
        pbar = tqdm(
            loader, desc=f"inverse-transition epoch {epoch + 1}/{args.epochs}", unit="batch"
        )
        for step, (before, after, target) in enumerate(pbar, start=1):
            global_batch += 1
            before = before.to(args.device, non_blocking=True)
            after = after.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            logits = model(before, after)
            loss = F.cross_entropy(logits.transpose(1, 2), target)
            (loss / args.grad_accum_steps).backward()

            if step % args.grad_accum_steps == 0 or step == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            prediction = logits.argmax(dim=-1)
            token_correct = int((prediction == target).sum().detach().cpu())
            token_count = int(target.numel())
            nonpad = target != pad_id
            nonpad_correct = int(((prediction == target) & nonpad).sum().detach().cpu())
            nonpad_count = int(nonpad.sum().detach().cpu())
            packet_correct = int((prediction == target).all(dim=-1).sum().detach().cpu())
            packet_count = int(target.shape[0])
            loss_window.append(float(loss.detach().cpu()))
            token_correct_window.append(token_correct)
            token_count_window.append(token_count)
            nonpad_correct_window.append(nonpad_correct)
            nonpad_count_window.append(nonpad_count)
            packet_correct_window.append(packet_correct)
            packet_count_window.append(packet_count)
            rolling_loss = sum(loss_window) / len(loss_window)
            pbar.set_postfix(loss=rolling_loss)

            should_print = step % 100 == 0 or step == len(loader)
            should_log_wandb = wandb_run is not None and (
                step % args.wandb_log_every == 0 or step == len(loader)
            )
            if should_print or should_log_wandb:
                metrics = {
                    "train/loss": rolling_loss,
                    "train/token_acc": sum(token_correct_window) / sum(token_count_window),
                    "train/nonpad_token_acc": (
                        sum(nonpad_correct_window) / sum(nonpad_count_window)
                        if sum(nonpad_count_window)
                        else 0.0
                    ),
                    "train/packet_acc": sum(packet_correct_window) / sum(packet_count_window),
                    "epoch": epoch + 1,
                    "batch": step,
                    "global_batch": global_batch,
                }
                if should_log_wandb:
                    wandb_run.log(metrics, step=global_batch)
                if should_print:
                    metrics_text = " ".join(
                        f"{key.removeprefix('train/')}={value:.4f}"
                        for key, value in metrics.items()
                        if key.startswith("train/")
                    )
                    pbar.write(f"epoch={epoch + 1} batch={step}/{len(loader)} {metrics_text}")

            if args.snapshot_every_batches and step % args.snapshot_every_batches == 0:
                snapshot_path = (
                    args.checkpoint_dir
                    / f"inverse_transition_epoch_{epoch + 1:03d}_batch_{step:06d}.pt"
                )
                save_checkpoint(
                    snapshot_path,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch + 1,
                    batch=step,
                    global_batch=global_batch,
                )
                pbar.write(f"saved snapshot: {snapshot_path}")

        checkpoint_path = args.checkpoint_dir / f"inverse_transition_epoch_{epoch + 1}.pt"
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            args=args,
            epoch=epoch + 1,
            batch=len(loader),
            global_batch=global_batch,
        )

    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
