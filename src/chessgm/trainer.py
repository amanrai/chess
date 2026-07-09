"""Training helpers for chess models."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from chessgm.data import VerifierGameStoreDataset
from chessgm.network import VerifierTransformer
from chessgm.tokenizer import VOCAB


@dataclass
class VerifierTrainConfig:
    data_dir: Path
    context_moves: int = 128
    batch_size: int = 32
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    model_dim: int = 256
    heads: int = 8
    layers: int = 6
    dropout: float = 0.0
    examples_per_epoch: int | None = None
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("checkpoints/verifier")
    log_window: int = 1000
    wandb: bool = False
    wandb_project: str = "chess-gm"
    wandb_run_name: str | None = None
    wandb_log_every: int = 10


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().detach().cpu())


def train_verifier(config: VerifierTrainConfig) -> VerifierTransformer:
    wandb_run = None
    if config.wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        )

    dataset = VerifierGameStoreDataset(
        config.data_dir,
        context_moves=config.context_moves,
        examples_per_epoch=config.examples_per_epoch,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )

    model = VerifierTransformer(
        vocab_size=len(VOCAB),
        move_expr=8,
        model_dim=config.model_dim,
        heads=config.heads,
        layers=config.layers,
        dropout=config.dropout,
        pad_id=0,
    ).to(config.device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"verifier epoch {epoch + 1}/{config.epochs}", unit="batch")
        running_loss = 0.0
        running_acc = 0.0
        loss_window: deque[float] = deque(maxlen=config.log_window)
        acc_window: deque[float] = deque(maxlen=config.log_window)
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_value = float(loss.detach().cpu())
            acc_value = accuracy(logits, y)
            running_loss += loss_value
            running_acc += acc_value
            loss_window.append(loss_value)
            acc_window.append(acc_value)
            global_step = epoch * len(loader) + step

            metrics = {
                "loss": loss_value,
                f"loss_last_{config.log_window}": sum(loss_window) / len(loss_window),
                "acc": acc_value,
                f"acc_last_{config.log_window}": sum(acc_window) / len(acc_window),
                "loss_epoch_avg": running_loss / step,
                "acc_epoch_avg": running_acc / step,
                "epoch": epoch + 1,
                "step": global_step,
            }
            pbar.set_postfix(
                loss_1000=metrics[f"loss_last_{config.log_window}"],
                acc_1000=metrics[f"acc_last_{config.log_window}"],
                loss=loss_value,
            )
            if wandb_run is not None and (step % config.wandb_log_every == 0):
                wandb_run.log(metrics, step=global_step)

        ckpt_path = config.checkpoint_dir / f"verifier_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": config.__dict__,
                "vocab": VOCAB,
                "epoch": epoch + 1,
            },
            ckpt_path,
        )
        if wandb_run is not None:
            wandb_run.log({"checkpoint_epoch": epoch + 1}, step=(epoch + 1) * len(loader))

    if wandb_run is not None:
        wandb_run.finish()
    return model
