#!/usr/bin/env python3
"""Train Q-encoder square-query probes for post-ply board occupancy."""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import uuid
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from chessgm.network_q import DiffThinkerBoardStateQueryMLP, QFormerPlyHistoryEncoder
from chessgm.tokenizer import VOCAB

NUM_SQUARES = 64
NUM_OCCUPANTS = 13
EMPTY = 0


OCCUPANT_NAMES = [
    "EMPTY",
    "WHITE_PAWN",
    "WHITE_KNIGHT",
    "WHITE_BISHOP",
    "WHITE_ROOK",
    "WHITE_QUEEN",
    "WHITE_KING",
    "BLACK_PAWN",
    "BLACK_KNIGHT",
    "BLACK_BISHOP",
    "BLACK_ROOK",
    "BLACK_QUEEN",
    "BLACK_KING",
]


def unpack_board_labels(packed: np.ndarray) -> np.ndarray:
    """Unpack uint8[..., 32] two-nibble board labels into uint8[..., 64]."""
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.shape[-1] != 32:
        raise ValueError(f"expected packed board last dim 32, got {packed.shape}")
    labels = np.empty((*packed.shape[:-1], 64), dtype=np.uint8)
    labels[..., 0::2] = packed & 0x0F
    labels[..., 1::2] = packed >> 4
    return labels


class BoardStateProbeDataset(Dataset):
    """Sample move-history prefixes and square-occupancy labels."""

    def __init__(
        self,
        data_dir: str | Path,
        board_state_dir: str | Path | None = None,
        context_plies: int = 128,
        squares_per_position: int = 16,
        examples_per_epoch: int | None = None,
        seed: int = 0,
        pad_id: int = 0,
        eval_all_squares: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.board_state_dir = Path(board_state_dir) if board_state_dir else self.data_dir / "board_state"
        self.moves = np.load(self.data_dir / "moves.npy", mmap_mode="r")
        self.offsets = np.load(self.data_dir / "offsets.npy", mmap_mode="r")
        self.boards = np.load(self.board_state_dir / "board_after_packed.npy", mmap_mode="r")
        self.samples = np.load(self.board_state_dir / "probe_samples.npy", mmap_mode="r")
        if self.samples.shape[1] != 2:
            raise ValueError(f"expected probe_samples [N, 2], got {self.samples.shape}")
        if len(self.boards) != len(self.moves):
            raise ValueError(f"board rows {len(self.boards)} do not match move rows {len(self.moves)}")
        if not 1 <= squares_per_position <= NUM_SQUARES:
            raise ValueError("squares_per_position must be in [1, 64]")
        self.context_plies = context_plies
        self.squares_per_position = NUM_SQUARES if eval_all_squares else squares_per_position
        self.examples_per_epoch = examples_per_epoch or len(self.samples)
        self.seed = seed
        self.pad_id = pad_id
        self.ply_expr = int(self.moves.shape[1])
        self.eval_all_squares = eval_all_squares

    def __len__(self) -> int:
        return self.examples_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_i = idx % len(self.samples)
        game_i, prefix_plies = (int(v) for v in self.samples[sample_i])
        start = int(self.offsets[game_i])
        end = int(self.offsets[game_i + 1])
        if prefix_plies < 1 or start + prefix_plies > end:
            raise ValueError(f"invalid probe sample game={game_i} prefix_plies={prefix_plies}")

        prefix = self.moves[start : start + prefix_plies]
        if len(prefix) >= self.context_plies:
            x = prefix[-self.context_plies :]
        else:
            pad_rows = np.full(
                (self.context_plies - len(prefix), self.ply_expr),
                self.pad_id,
                dtype=np.uint16,
            )
            x = np.concatenate([pad_rows, prefix], axis=0)

        board_row = start + prefix_plies - 1
        labels64 = unpack_board_labels(self.boards[board_row])
        if self.eval_all_squares:
            square_ids = np.arange(NUM_SQUARES, dtype=np.int64)
        else:
            rng = random.Random(self.seed + idx)
            square_ids = np.asarray(rng.sample(range(NUM_SQUARES), self.squares_per_position), dtype=np.int64)
        labels = labels64[square_ids].astype(np.int64)
        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(square_ids),
            torch.from_numpy(labels),
        )


class QBoardStateProbeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ply_expr: int = 8,
        model_dim: int = 256,
        heads: int = 8,
        history_layers: int = 4,
        q_layers: int = 2,
        num_queries: int = 16,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.encoder = QFormerPlyHistoryEncoder(
            vocab_size=vocab_size,
            ply_expr=ply_expr,
            model_dim=model_dim,
            heads=heads,
            history_layers=history_layers,
            q_layers=q_layers,
            num_queries=num_queries,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.board_head = DiffThinkerBoardStateQueryMLP(model_dim=model_dim, dropout=dropout)

    def forward(self, x_ids: torch.Tensor, square_ids: torch.Tensor) -> torch.Tensor:
        q = self.encoder(x_ids)
        return self.board_head(q, square_ids)


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
    try:
        api_key = os.environ.get("WANDB_API_KEY") or read_dotenv_key()
        import wandb

        if api_key:
            wandb.login(key=api_key)
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        )
    except Exception as exc:
        print(f"W&B initialization failed; continuing without W&B logging: {exc}")
        return None


def checkpoint_run_id(wandb_run) -> str:
    if wandb_run is not None and getattr(wandb_run, "name", None):
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", wandb_run.name).strip("-") or wandb_run.id
    return uuid.uuid4().hex[:8]


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    batch: int,
    global_batch: int,
    run_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "batch": batch,
            "global_batch": global_batch,
            "run_id": run_id,
        },
        path,
    )


def load_encoder_checkpoint(model: QBoardStateProbeTransformer, path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state.items() if k.startswith("encoder.") and k in model_state and model_state[k].shape == v.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"loaded encoder weights from {path}: compatible={len(compatible)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def precision_recall(tp: int, pred_pos: int, pos: int) -> tuple[float, float]:
    return (tp / pred_pos if pred_pos else 0.0, tp / pos if pos else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "lumbras" / "verifier")
    parser.add_argument("--board-state-dir", type=Path, default=None)
    parser.add_argument("--context-plies", type=int, default=128)
    parser.add_argument("--squares-per-position", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
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
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional q-probe checkpoint to initialize the shared encoder")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "board_state_q_probe")
    parser.add_argument("--snapshot-every-batches", type=int, default=5000, help="Save an in-epoch snapshot every N batches; 0 disables")
    parser.add_argument("--log-window", type=int, default=1000)
    parser.add_argument("--print-every-batches", type=int, default=25)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="chess-gm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-log-every", type=int, default=100)
    args = parser.parse_args()

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.snapshot_every_batches < 0:
        raise ValueError("--snapshot-every-batches must be >= 0")
    if args.print_every_batches < 0:
        raise ValueError("--print-every-batches must be >= 0")
    if not 1 <= args.squares_per_position <= NUM_SQUARES:
        raise ValueError("--squares-per-position must be in [1, 64]")

    wandb_run = init_wandb(args)
    run_id = checkpoint_run_id(wandb_run)
    print(f"checkpoint run id: {run_id}")

    dataset = BoardStateProbeDataset(
        args.data_dir,
        board_state_dir=args.board_state_dir,
        context_plies=args.context_plies,
        squares_per_position=args.squares_per_position,
        examples_per_epoch=args.examples_per_epoch,
        seed=0,
    )
    print(
        f"dataset: samples={len(dataset.samples):,} examples_per_epoch={len(dataset):,} "
        f"context_plies={args.context_plies} squares_per_position={args.squares_per_position}"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = QBoardStateProbeTransformer(
        vocab_size=len(VOCAB),
        ply_expr=dataset.ply_expr,
        model_dim=args.model_dim,
        heads=args.heads,
        history_layers=args.history_layers,
        q_layers=args.q_layers,
        num_queries=args.num_queries,
        dropout=args.dropout,
    )
    if args.init_checkpoint is not None:
        load_encoder_checkpoint(model, args.init_checkpoint)
    model = model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.board_head.parameters())
    head_trainable_params = sum(p.numel() for p in model.board_head.parameters() if p.requires_grad)
    print(
        f"parameters: total={total_params:,} trainable={trainable_params:,} "
        f"encoder={encoder_params:,} encoder_trainable={encoder_trainable_params:,} "
        f"board_head={head_params:,} board_head_trainable={head_trainable_params:,}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_window: deque[float] = deque(maxlen=args.log_window)
    acc_window: deque[float] = deque(maxlen=args.log_window)
    occupied_acc_window: deque[float] = deque(maxlen=args.log_window)
    occupied_precision_window: deque[float] = deque(maxlen=args.log_window)
    non_empty_recall_window: deque[float] = deque(maxlen=args.log_window)
    global_batch = 0

    for epoch in range(args.epochs):
        model.train()
        class_correct = torch.zeros(NUM_OCCUPANTS, dtype=torch.long)
        class_total = torch.zeros(NUM_OCCUPANTS, dtype=torch.long)
        class_pred_total = torch.zeros(NUM_OCCUPANTS, dtype=torch.long)
        pbar = tqdm(loader, desc=f"board-state q-probe epoch {epoch + 1}/{args.epochs}", unit="batch")
        for step, (x, square_ids, labels) in enumerate(pbar, start=1):
            global_batch += 1
            x = x.to(args.device, non_blocking=True)
            square_ids = square_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            logits = model(x, square_ids)
            loss = F.cross_entropy(logits.reshape(-1, NUM_OCCUPANTS), labels.reshape(-1))
            (loss / args.grad_accum_steps).backward()
            if step % args.grad_accum_steps == 0 or step == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            pred = logits.argmax(dim=-1)
            correct = pred.eq(labels)
            occupied = labels.ne(EMPTY)
            pred_occupied = pred.ne(EMPTY)
            tp_occupied = (pred_occupied & occupied).sum().item()
            acc = correct.float().mean().item()
            occupied_acc = correct[occupied].float().mean().item() if occupied.any() else 0.0
            occupied_precision = tp_occupied / pred_occupied.sum().item() if pred_occupied.any() else 0.0
            non_empty_recall = tp_occupied / occupied.sum().item() if occupied.any() else 0.0
            loss_window.append(float(loss.detach().cpu()))
            acc_window.append(acc)
            occupied_acc_window.append(occupied_acc)
            occupied_precision_window.append(occupied_precision)
            non_empty_recall_window.append(non_empty_recall)
            batch_class_total = torch.bincount(labels.detach().cpu().reshape(-1), minlength=NUM_OCCUPANTS)
            batch_class_pred_total = torch.bincount(pred.detach().cpu().reshape(-1), minlength=NUM_OCCUPANTS)
            batch_class_correct = torch.bincount(labels.detach().cpu().reshape(-1)[correct.detach().cpu().reshape(-1)], minlength=NUM_OCCUPANTS)
            class_total += batch_class_total
            class_pred_total += batch_class_pred_total
            class_correct += batch_class_correct

            rolling_loss = sum(loss_window) / len(loss_window)
            pbar.set_postfix(loss=rolling_loss, acc=sum(acc_window) / len(acc_window))

            if wandb_run is not None and (step % args.wandb_log_every == 0 or step == len(loader)):
                occupied_total = int((class_total[1:]).sum())
                occupied_correct = int((class_correct[1:]).sum())
                log_payload = {
                    "train/loss": rolling_loss,
                    "train/square_acc": sum(acc_window) / len(acc_window),
                    "train/occupied_square_acc": sum(occupied_acc_window) / len(occupied_acc_window),
                    "train/occupied_precision": sum(occupied_precision_window) / len(occupied_precision_window),
                    "train/non_empty_recall": sum(non_empty_recall_window) / len(non_empty_recall_window),
                    "train/epoch_occupied_acc": occupied_correct / occupied_total if occupied_total else 0.0,
                    "epoch": epoch + 1,
                    "batch": step,
                }
                for cls, name in enumerate(OCCUPANT_NAMES):
                    total = int(class_total[cls])
                    pred_total = int(class_pred_total[cls])
                    log_payload[f"class/{name}/acc"] = int(class_correct[cls]) / total if total else 0.0
                    log_payload[f"class/{name}/precision"] = int(class_correct[cls]) / pred_total if pred_total else 0.0
                    log_payload[f"class/{name}/n"] = total
                    log_payload[f"class/{name}/pred_n"] = pred_total
                wandb_run.log(log_payload, step=(epoch * len(loader)) + step)

            if step == 1 or (args.print_every_batches and step % args.print_every_batches == 0) or step == len(loader):
                occupied_total = int(class_total[1:].sum())
                occupied_pred_total = int(class_pred_total[1:].sum())
                occupied_correct = int(class_correct[1:].sum())
                empty_total = int(class_total[EMPTY])
                empty_pred_total = int(class_pred_total[EMPTY])
                empty_correct = int(class_correct[EMPTY])
                print(
                    f"\nepoch={epoch + 1} batch={step}/{len(loader)} "
                    f"loss={rolling_loss:.4f} square_acc={sum(acc_window) / len(acc_window):.4f} "
                    f"empty_acc={(empty_correct / empty_total if empty_total else 0.0):.4f} "
                    f"empty_precision={(empty_correct / empty_pred_total if empty_pred_total else 0.0):.4f} "
                    f"occupied_acc={(occupied_correct / occupied_total if occupied_total else 0.0):.4f} "
                    f"occupied_precision={(occupied_correct / occupied_pred_total if occupied_pred_total else 0.0):.4f} "
                    f"non_empty_recall={sum(non_empty_recall_window) / len(non_empty_recall_window):.4f} "
                    f"occupied_n={occupied_total} occupied_pred_n={occupied_pred_total} empty_n={empty_total} empty_pred_n={empty_pred_total}"
                )
                rows = []
                for cls, name in enumerate(OCCUPANT_NAMES):
                    total = int(class_total[cls])
                    pred_total = int(class_pred_total[cls])
                    if total or pred_total:
                        recall = int(class_correct[cls]) / total if total else 0.0
                        precision = int(class_correct[cls]) / pred_total if pred_total else 0.0
                        rows.append(f"{name}:p={precision:.3f},r={recall:.3f},n={total},pred={pred_total}")
                pbar.write("class_precision_recall " + " ".join(rows))

            if args.snapshot_every_batches and step % args.snapshot_every_batches == 0:
                snapshot_path = args.checkpoint_dir / f"board_state_q_probe_{run_id}_epoch_{epoch + 1:03d}_batch_{step:06d}.pt"
                save_checkpoint(
                    snapshot_path,
                    model=model,
                    opt=opt,
                    args=args,
                    epoch=epoch + 1,
                    batch=step,
                    global_batch=global_batch,
                    run_id=run_id,
                )
                pbar.write(f"saved snapshot: {snapshot_path}")

        ckpt_path = args.checkpoint_dir / f"board_state_q_probe_{run_id}_epoch_{epoch + 1}.pt"
        save_checkpoint(
            ckpt_path,
            model=model,
            opt=opt,
            args=args,
            epoch=epoch + 1,
            batch=len(loader),
            global_batch=global_batch,
            run_id=run_id,
        )
        print(f"saved checkpoint: {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
