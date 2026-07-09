"""Training helpers for chess models."""
from __future__ import annotations

from dataclasses import dataclass
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


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().detach().cpu())


def train_verifier(config: VerifierTrainConfig) -> VerifierTransformer:
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
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += float(loss.detach().cpu())
            running_acc += accuracy(logits, y)
            pbar.set_postfix(loss=running_loss / step, acc=running_acc / step)

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

    return model
