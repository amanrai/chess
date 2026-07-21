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

from chessgm.network_q import DiffThinkerBoardStateQueryMLP, DiffThinkerMLP, QFormerPlyHistoryEncoder
from chessgm.tokenizer import TOKEN_TO_ID, VOCAB

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
        self.check_id = TOKEN_TO_ID["CHECK"]
        self.mate_id = TOKEN_TO_ID["MATE"]
        self.leak_label_ids = {self.check_id, self.mate_id}

    def __len__(self) -> int:
        return self.examples_per_epoch

    def strip_check_mate_tokens(self, prefix: np.ndarray) -> np.ndarray:
        stripped = np.full(prefix.shape, self.pad_id, dtype=prefix.dtype)
        for row_i, row in enumerate(prefix):
            kept = [int(token_id) for token_id in row if int(token_id) not in self.leak_label_ids]
            kept = kept[: self.ply_expr]
            stripped[row_i, : len(kept)] = kept
        return stripped

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_i = idx % len(self.samples)
        game_i, prefix_plies = (int(v) for v in self.samples[sample_i])
        start = int(self.offsets[game_i])
        end = int(self.offsets[game_i + 1])
        if prefix_plies < 1 or start + prefix_plies > end:
            raise ValueError(f"invalid probe sample game={game_i} prefix_plies={prefix_plies}")

        prefix = self.moves[start : start + prefix_plies]
        target_ply = prefix[-1]
        check = int((target_ply == self.check_id).any() or (target_ply == self.mate_id).any())
        mate = int((target_ply == self.mate_id).any())
        next_turn = 1 if prefix_plies % 2 == 1 else 0
        prefix_x = self.strip_check_mate_tokens(prefix)
        if len(prefix_x) >= self.context_plies:
            x = prefix_x[-self.context_plies :]
        else:
            pad_rows = np.full(
                (self.context_plies - len(prefix), self.ply_expr),
                self.pad_id,
                dtype=np.uint16,
            )
            x = np.concatenate([pad_rows, prefix_x], axis=0)

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
            torch.tensor(check, dtype=torch.long),
            torch.tensor(mate, dtype=torch.long),
            torch.tensor(next_turn, dtype=torch.long),
            torch.tensor(prefix_plies, dtype=torch.long),
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
        self.check_head = DiffThinkerMLP(model_dim=model_dim, num_outputs=2, dropout=dropout)
        self.mate_head = DiffThinkerMLP(model_dim=model_dim, num_outputs=2, dropout=dropout)
        self.turn_head = DiffThinkerMLP(model_dim=model_dim, num_outputs=2, dropout=dropout)

    def forward(self, x_ids: torch.Tensor, square_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.encoder(x_ids)
        return self.board_head(q, square_ids), self.check_head(q), self.mate_head(q), self.turn_head(q)


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


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def sep() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)) + " |"

    return "\n".join([sep(), fmt_row(headers), sep(), *(fmt_row(row) for row in rows), sep()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "lumbras" / "verifier")
    parser.add_argument("--board-state-dir", type=Path, default=None)
    parser.add_argument("--context-plies", type=int, default=128)
    parser.add_argument("--squares-per-position", type=int, default=16)
    parser.add_argument("--bucket-plies", type=int, default=25, help="Bucket size for per-ply-range board/probe logs")
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
    if args.bucket_plies < 1:
        raise ValueError("--bucket-plies must be >= 1")

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
    board_loss_window: deque[float] = deque(maxlen=args.log_window)
    check_loss_window: deque[float] = deque(maxlen=args.log_window)
    mate_loss_window: deque[float] = deque(maxlen=args.log_window)
    turn_loss_window: deque[float] = deque(maxlen=args.log_window)
    acc_window: deque[float] = deque(maxlen=args.log_window)
    occupied_acc_window: deque[float] = deque(maxlen=args.log_window)
    occupied_precision_window: deque[float] = deque(maxlen=args.log_window)
    non_empty_recall_window: deque[float] = deque(maxlen=args.log_window)
    check_acc_window: deque[float] = deque(maxlen=args.log_window)
    mate_acc_window: deque[float] = deque(maxlen=args.log_window)
    turn_acc_window: deque[float] = deque(maxlen=args.log_window)
    check_tp_window: deque[int] = deque(maxlen=args.log_window)
    check_pos_window: deque[int] = deque(maxlen=args.log_window)
    check_pred_pos_window: deque[int] = deque(maxlen=args.log_window)
    mate_tp_window: deque[int] = deque(maxlen=args.log_window)
    mate_pos_window: deque[int] = deque(maxlen=args.log_window)
    mate_pred_pos_window: deque[int] = deque(maxlen=args.log_window)
    black_tp_window: deque[int] = deque(maxlen=args.log_window)
    black_pos_window: deque[int] = deque(maxlen=args.log_window)
    black_pred_pos_window: deque[int] = deque(maxlen=args.log_window)
    global_batch = 0

    for epoch in range(args.epochs):
        model.train()
        class_correct = torch.zeros(NUM_OCCUPANTS, dtype=torch.long)
        class_total = torch.zeros(NUM_OCCUPANTS, dtype=torch.long)
        class_pred_total = torch.zeros(NUM_OCCUPANTS, dtype=torch.long)
        bucket_stats: dict[int, dict[str, int]] = {}
        pbar = tqdm(loader, desc=f"board-state q-probe epoch {epoch + 1}/{args.epochs}", unit="batch")
        for step, (x, square_ids, labels, check_y, mate_y, turn_y, probe_ply) in enumerate(pbar, start=1):
            global_batch += 1
            x = x.to(args.device, non_blocking=True)
            square_ids = square_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            check_y = check_y.to(args.device, non_blocking=True)
            mate_y = mate_y.to(args.device, non_blocking=True)
            turn_y = turn_y.to(args.device, non_blocking=True)
            logits, check_logits, mate_logits, turn_logits = model(x, square_ids)
            board_loss = F.cross_entropy(logits.reshape(-1, NUM_OCCUPANTS), labels.reshape(-1))
            check_loss = F.cross_entropy(check_logits, check_y)
            mate_loss = F.cross_entropy(mate_logits, mate_y)
            turn_loss = F.cross_entropy(turn_logits, turn_y)
            loss = board_loss + check_loss + mate_loss + turn_loss
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
            board_loss_window.append(float(board_loss.detach().cpu()))
            check_loss_window.append(float(check_loss.detach().cpu()))
            mate_loss_window.append(float(mate_loss.detach().cpu()))
            turn_loss_window.append(float(turn_loss.detach().cpu()))
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

            check_pred = check_logits.argmax(dim=-1)
            mate_pred = mate_logits.argmax(dim=-1)
            turn_pred = turn_logits.argmax(dim=-1)
            check_acc_window.append((check_pred == check_y).float().mean().item())
            mate_acc_window.append((mate_pred == mate_y).float().mean().item())
            turn_acc_window.append((turn_pred == turn_y).float().mean().item())
            check_tp_window.append(int(((check_pred == 1) & (check_y == 1)).sum().detach().cpu()))
            check_pos_window.append(int((check_y == 1).sum().detach().cpu()))
            check_pred_pos_window.append(int((check_pred == 1).sum().detach().cpu()))
            mate_tp_window.append(int(((mate_pred == 1) & (mate_y == 1)).sum().detach().cpu()))
            mate_pos_window.append(int((mate_y == 1).sum().detach().cpu()))
            mate_pred_pos_window.append(int((mate_pred == 1).sum().detach().cpu()))
            black_tp_window.append(int(((turn_pred == 1) & (turn_y == 1)).sum().detach().cpu()))
            black_pos_window.append(int((turn_y == 1).sum().detach().cpu()))
            black_pred_pos_window.append(int((turn_pred == 1).sum().detach().cpu()))

            sample_square_correct = correct.sum(dim=1).detach().cpu().tolist()
            sample_occupied = occupied.sum(dim=1).detach().cpu().tolist()
            sample_pred_occupied = pred_occupied.sum(dim=1).detach().cpu().tolist()
            sample_occupied_correct = (correct & occupied).sum(dim=1).detach().cpu().tolist()
            sample_check_y = check_y.detach().cpu().tolist()
            sample_check_pred = check_pred.detach().cpu().tolist()
            sample_mate_y = mate_y.detach().cpu().tolist()
            sample_mate_pred = mate_pred.detach().cpu().tolist()
            sample_turn_y = turn_y.detach().cpu().tolist()
            sample_turn_pred = turn_pred.detach().cpu().tolist()
            for i, bucket in enumerate(((probe_ply - 1) // args.bucket_plies).tolist()):
                bucket = int(bucket)
                stats = bucket_stats.setdefault(
                    bucket,
                    {
                        "n": 0,
                        "squares": 0,
                        "square_correct": 0,
                        "occupied": 0,
                        "pred_occupied": 0,
                        "occupied_correct": 0,
                        "check_pos": 0,
                        "check_pred_pos": 0,
                        "check_tp": 0,
                        "mate_pos": 0,
                        "mate_pred_pos": 0,
                        "mate_tp": 0,
                        "black_pos": 0,
                        "black_pred_pos": 0,
                        "black_tp": 0,
                        "turn_correct": 0,
                    },
                )
                stats["n"] += 1
                stats["squares"] += int(labels.shape[1])
                stats["square_correct"] += int(sample_square_correct[i])
                stats["occupied"] += int(sample_occupied[i])
                stats["pred_occupied"] += int(sample_pred_occupied[i])
                stats["occupied_correct"] += int(sample_occupied_correct[i])
                stats["check_pos"] += int(sample_check_y[i] == 1)
                stats["check_pred_pos"] += int(sample_check_pred[i] == 1)
                stats["check_tp"] += int(sample_check_y[i] == 1 and sample_check_pred[i] == 1)
                stats["mate_pos"] += int(sample_mate_y[i] == 1)
                stats["mate_pred_pos"] += int(sample_mate_pred[i] == 1)
                stats["mate_tp"] += int(sample_mate_y[i] == 1 and sample_mate_pred[i] == 1)
                stats["black_pos"] += int(sample_turn_y[i] == 1)
                stats["black_pred_pos"] += int(sample_turn_pred[i] == 1)
                stats["black_tp"] += int(sample_turn_y[i] == 1 and sample_turn_pred[i] == 1)
                stats["turn_correct"] += int(sample_turn_y[i] == sample_turn_pred[i])

            rolling_loss = sum(loss_window) / len(loss_window)
            pbar.set_postfix(loss=rolling_loss, square_acc=sum(acc_window) / len(acc_window))

            if wandb_run is not None and (step % args.wandb_log_every == 0 or step == len(loader)):
                occupied_total = int((class_total[1:]).sum())
                occupied_correct = int((class_correct[1:]).sum())
                check_pos = sum(check_pos_window)
                check_pred_pos = sum(check_pred_pos_window)
                mate_pos = sum(mate_pos_window)
                mate_pred_pos = sum(mate_pred_pos_window)
                black_pos = sum(black_pos_window)
                black_pred_pos = sum(black_pred_pos_window)
                black_tp = sum(black_tp_window)
                log_payload = {
                    "train/loss": rolling_loss,
                    "train/board_loss": sum(board_loss_window) / len(board_loss_window),
                    "train/check_loss": sum(check_loss_window) / len(check_loss_window),
                    "train/mate_loss": sum(mate_loss_window) / len(mate_loss_window),
                    "train/turn_loss": sum(turn_loss_window) / len(turn_loss_window),
                    "train/square_acc": sum(acc_window) / len(acc_window),
                    "train/occupied_square_acc": sum(occupied_acc_window) / len(occupied_acc_window),
                    "train/occupied_precision": sum(occupied_precision_window) / len(occupied_precision_window),
                    "train/non_empty_recall": sum(non_empty_recall_window) / len(non_empty_recall_window),
                    "train/epoch_occupied_acc": occupied_correct / occupied_total if occupied_total else 0.0,
                    "train/check_acc": sum(check_acc_window) / len(check_acc_window),
                    "train/mate_acc": sum(mate_acc_window) / len(mate_acc_window),
                    "train/turn_acc": sum(turn_acc_window) / len(turn_acc_window),
                    "train/p_check": sum(check_tp_window) / check_pred_pos if check_pred_pos else 0.0,
                    "train/r_check": sum(check_tp_window) / check_pos if check_pos else 0.0,
                    "train/p_mate": sum(mate_tp_window) / mate_pred_pos if mate_pred_pos else 0.0,
                    "train/r_mate": sum(mate_tp_window) / mate_pos if mate_pos else 0.0,
                    "train/p_black": black_tp / black_pred_pos if black_pred_pos else 0.0,
                    "train/r_black": black_tp / black_pos if black_pos else 0.0,
                    "epoch": epoch + 1,
                    "batch": step,
                }
                for cls, name in enumerate(OCCUPANT_NAMES):
                    total = int(class_total[cls])
                    pred_total = int(class_pred_total[cls])
                    log_payload[f"class/{name}/recall"] = int(class_correct[cls]) / total if total else 0.0
                    log_payload[f"class/{name}/precision"] = int(class_correct[cls]) / pred_total if pred_total else 0.0
                    log_payload[f"class/{name}/n"] = total
                    log_payload[f"class/{name}/pred_n"] = pred_total
                for bucket, stats in bucket_stats.items():
                    lo = bucket * args.bucket_plies + 1
                    hi = (bucket + 1) * args.bucket_plies
                    prefix = f"bucket/{lo}_{hi}"
                    log_payload[f"{prefix}/square_acc"] = stats["square_correct"] / stats["squares"] if stats["squares"] else 0.0
                    log_payload[f"{prefix}/occupied_precision"] = stats["occupied_correct"] / stats["pred_occupied"] if stats["pred_occupied"] else 0.0
                    log_payload[f"{prefix}/occupied_recall"] = stats["occupied_correct"] / stats["occupied"] if stats["occupied"] else 0.0
                    log_payload[f"{prefix}/p_check"] = stats["check_tp"] / stats["check_pred_pos"] if stats["check_pred_pos"] else 0.0
                    log_payload[f"{prefix}/r_check"] = stats["check_tp"] / stats["check_pos"] if stats["check_pos"] else 0.0
                    log_payload[f"{prefix}/p_mate"] = stats["mate_tp"] / stats["mate_pred_pos"] if stats["mate_pred_pos"] else 0.0
                    log_payload[f"{prefix}/r_mate"] = stats["mate_tp"] / stats["mate_pos"] if stats["mate_pos"] else 0.0
                    log_payload[f"{prefix}/p_black"] = stats["black_tp"] / stats["black_pred_pos"] if stats["black_pred_pos"] else 0.0
                    log_payload[f"{prefix}/r_black"] = stats["black_tp"] / stats["black_pos"] if stats["black_pos"] else 0.0
                    log_payload[f"{prefix}/turn_acc"] = stats["turn_correct"] / stats["n"] if stats["n"] else 0.0
                    log_payload[f"{prefix}/n"] = stats["n"]
                wandb_run.log(log_payload, step=(epoch * len(loader)) + step)

            if step == 1 or (args.print_every_batches and step % args.print_every_batches == 0) or step == len(loader):
                occupied_total = int(class_total[1:].sum())
                occupied_pred_total = int(class_pred_total[1:].sum())
                occupied_correct = int(class_correct[1:].sum())
                empty_total = int(class_total[EMPTY])
                empty_pred_total = int(class_pred_total[EMPTY])
                empty_correct = int(class_correct[EMPTY])
                check_pos = sum(check_pos_window)
                check_pred_pos = sum(check_pred_pos_window)
                mate_pos = sum(mate_pos_window)
                mate_pred_pos = sum(mate_pred_pos_window)
                black_pos = sum(black_pos_window)
                black_pred_pos = sum(black_pred_pos_window)
                black_tp = sum(black_tp_window)
                summary_table = format_table(
                    ["metric", "value"],
                    [
                        ["loss", f"{rolling_loss:.4f}"],
                        ["board_loss", f"{sum(board_loss_window) / len(board_loss_window):.4f}"],
                        ["check_loss", f"{sum(check_loss_window) / len(check_loss_window):.4f}"],
                        ["mate_loss", f"{sum(mate_loss_window) / len(mate_loss_window):.4f}"],
                        ["turn_loss", f"{sum(turn_loss_window) / len(turn_loss_window):.4f}"],
                        ["square_acc", f"{sum(acc_window) / len(acc_window):.4f}"],
                        ["empty_precision", f"{(empty_correct / empty_pred_total if empty_pred_total else 0.0):.4f}"],
                        ["empty_recall", f"{(empty_correct / empty_total if empty_total else 0.0):.4f}"],
                        ["occupied_precision", f"{(occupied_correct / occupied_pred_total if occupied_pred_total else 0.0):.4f}"],
                        ["occupied_recall", f"{(occupied_correct / occupied_total if occupied_total else 0.0):.4f}"],
                        ["check_acc", f"{sum(check_acc_window) / len(check_acc_window):.4f}"],
                        ["p_check", f"{(sum(check_tp_window) / check_pred_pos if check_pred_pos else 0.0):.4f}"],
                        ["r_check", f"{(sum(check_tp_window) / check_pos if check_pos else 0.0):.4f}"],
                        ["mate_acc", f"{sum(mate_acc_window) / len(mate_acc_window):.4f}"],
                        ["p_mate", f"{(sum(mate_tp_window) / mate_pred_pos if mate_pred_pos else 0.0):.4f}"],
                        ["r_mate", f"{(sum(mate_tp_window) / mate_pos if mate_pos else 0.0):.4f}"],
                        ["turn_acc", f"{sum(turn_acc_window) / len(turn_acc_window):.4f}"],
                        ["p_black", f"{(black_tp / black_pred_pos if black_pred_pos else 0.0):.4f}"],
                        ["r_black", f"{(black_tp / black_pos if black_pos else 0.0):.4f}"],
                        ["occupied_n", str(occupied_total)],
                        ["occupied_pred_n", str(occupied_pred_total)],
                        ["empty_n", str(empty_total)],
                        ["empty_pred_n", str(empty_pred_total)],
                    ],
                )
                class_rows = []
                for cls, name in enumerate(OCCUPANT_NAMES):
                    total = int(class_total[cls])
                    pred_total = int(class_pred_total[cls])
                    correct_cls = int(class_correct[cls])
                    if total or pred_total:
                        recall = correct_cls / total if total else 0.0
                        precision = correct_cls / pred_total if pred_total else 0.0
                        class_rows.append([name, f"{precision:.3f}", f"{recall:.3f}", str(correct_cls), str(total), str(pred_total)])
                class_table = format_table(["class", "precision", "recall", "correct", "n", "pred_n"], class_rows)
                bucket_rows = []
                for bucket in sorted(bucket_stats):
                    stats = bucket_stats[bucket]
                    bucket_rows.append(
                        [
                            f"{bucket * args.bucket_plies + 1}-{(bucket + 1) * args.bucket_plies}",
                            f"{(stats['square_correct'] / stats['squares'] if stats['squares'] else 0.0):.3f}",
                            f"{(stats['occupied_correct'] / stats['pred_occupied'] if stats['pred_occupied'] else 0.0):.3f}",
                            f"{(stats['occupied_correct'] / stats['occupied'] if stats['occupied'] else 0.0):.3f}",
                            f"{(stats['check_tp'] / stats['check_pred_pos'] if stats['check_pred_pos'] else 0.0):.3f}",
                            f"{(stats['check_tp'] / stats['check_pos'] if stats['check_pos'] else 0.0):.3f}",
                            str(stats['check_pos']),
                            str(stats['check_pred_pos']),
                            f"{(stats['mate_tp'] / stats['mate_pred_pos'] if stats['mate_pred_pos'] else 0.0):.3f}",
                            f"{(stats['mate_tp'] / stats['mate_pos'] if stats['mate_pos'] else 0.0):.3f}",
                            str(stats['mate_pos']),
                            str(stats['mate_pred_pos']),
                            f"{(stats['black_tp'] / stats['black_pred_pos'] if stats['black_pred_pos'] else 0.0):.3f}",
                            f"{(stats['black_tp'] / stats['black_pos'] if stats['black_pos'] else 0.0):.3f}",
                            f"{(stats['turn_correct'] / stats['n'] if stats['n'] else 0.0):.3f}",
                            str(stats['n']),
                        ]
                    )
                bucket_table = format_table(
                    ["plies", "sq_acc", "occ_p", "occ_r", "p_chk", "r_chk", "chk+", "chk_pred", "p_mate", "r_mate", "mate+", "mate_pred", "p_blk", "r_blk", "turn_a", "n"],
                    bucket_rows,
                )
                pbar.write(
                    f"\nepoch={epoch + 1} batch={step}/{len(loader)}\n"
                    f"\nsummary\n{summary_table}\n"
                    f"\noccupant classes\n{class_table}\n"
                    f"\nply buckets\n{bucket_table}"
                )
                bucket_stats.clear()

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
