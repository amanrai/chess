#!/usr/bin/env python3
"""Train Q-encoder probes for basic prefix facts: check and next turn."""
from __future__ import annotations

import argparse
import os
import random
import sys
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

from chessgm.network_q import QFormerPlyHistoryEncoder
from chessgm.tokenizer import TOKEN_TO_ID, VOCAB


class PlyProbeDataset(Dataset):
    """Sample a random prefix and label final-ply check, mate, and next turn."""

    def __init__(
        self,
        root: str | Path,
        context_plies: int = 128,
        max_probe_plies: int | None = None,
        min_game_plies: int | None = None,
        max_game_plies: int | None = None,
        examples_per_epoch: int | None = None,
        seed: int = 0,
        pad_id: int = 0,
    ):
        self.root = Path(root)
        self.moves = np.load(self.root / "moves.npy", mmap_mode="r")
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self.context_plies = context_plies
        self.max_probe_plies = max_probe_plies
        self.seed = seed
        self.pad_id = pad_id
        self.ply_expr = int(self.moves.shape[1])
        self.check_id = TOKEN_TO_ID["CHECK"]
        self.mate_id = TOKEN_TO_ID["MATE"]
        self.leak_label_ids = {self.check_id, self.mate_id}

        game_lengths_plies = np.diff(self.offsets)
        valid_mask = game_lengths_plies > 0
        if min_game_plies is not None:
            valid_mask &= game_lengths_plies >= min_game_plies
        if max_game_plies is not None:
            valid_mask &= game_lengths_plies <= max_game_plies
        if max_probe_plies is not None:
            valid_mask &= np.minimum(game_lengths_plies, max_probe_plies) > 0
        self.game_indices = np.flatnonzero(valid_mask).astype(np.int64)
        if len(self.game_indices) == 0:
            raise ValueError("no games left after ply-probe filtering")
        self.examples_per_epoch = examples_per_epoch or len(self.game_indices)

    def __len__(self) -> int:
        return self.examples_per_epoch

    def strip_check_mate_tokens(self, prefix: np.ndarray) -> np.ndarray:
        """Remove CHECK/MATE tokens from each ply and shift remaining tokens left.

        Replacing CHECK/MATE in-place with PAD would leak the suffix position, e.g.
        `... TO_h5 PAD EOM ...`. This instead produces normal packet structure:
        `... TO_h5 EOM PAD ...`.
        """
        stripped = np.full(prefix.shape, self.pad_id, dtype=prefix.dtype)
        for row_i, row in enumerate(prefix):
            kept = [int(token_id) for token_id in row if int(token_id) not in self.leak_label_ids]
            kept = kept[: self.ply_expr]
            stripped[row_i, : len(kept)] = kept
        return stripped

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        game_i = int(self.game_indices[rng.randrange(len(self.game_indices))])
        start = int(self.offsets[game_i])
        end = int(self.offsets[game_i + 1])
        num_plies = end - start
        max_t = min(num_plies, self.max_probe_plies) if self.max_probe_plies else num_plies
        t = rng.randint(1, max_t)

        prefix = self.moves[start : start + t]
        target_ply = prefix[-1]
        check = int((target_ply == self.check_id).any() or (target_ply == self.mate_id).any())
        mate = int((target_ply == self.mate_id).any())
        next_turn = 1 if t % 2 == 1 else 0  # 0=white, 1=black; after white ply, black moves.

        # Prevent label leakage: CHECK/MATE tokens are targets, never inputs.
        # Remove them and shift the remaining tokens left instead of leaving a PAD hole.
        prefix_x = self.strip_check_mate_tokens(prefix)

        if len(prefix_x) >= self.context_plies:
            x = prefix_x[-self.context_plies :]
        else:
            pad_rows = np.full(
                (self.context_plies - len(prefix_x), self.ply_expr),
                self.pad_id,
                dtype=np.uint16,
            )
            x = np.concatenate([pad_rows, prefix_x], axis=0)

        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.tensor(check, dtype=torch.long),
            torch.tensor(mate, dtype=torch.long),
            torch.tensor(next_turn, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
        )


class QPlyProbeTransformer(nn.Module):
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
        self.check_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2),
        )
        self.mate_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2),
        )
        self.turn_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2),
        )

    def forward(self, x_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.encoder(x_ids)
        pooled = q.mean(dim=1)
        return self.check_head(pooled), self.mate_head(pooled), self.turn_head(pooled)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().detach().cpu())


def linear_decay(start: float, end: float, step: int, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    progress = min(max(step, 0) / decay_steps, 1.0)
    return start + (end - start) * progress


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
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )


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


def load_encoder_checkpoint(model: QPlyProbeTransformer, path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "lumbras" / "verifier")
    parser.add_argument("--context-plies", type=int, default=128)
    parser.add_argument("--max-probe-plies", type=int, default=250, help="Sample probe prefix length uniformly from 1..min(game plies, this)")
    parser.add_argument("--bucket-plies", type=int, default=25, help="Bucket size for per-ply-range turn logs")
    parser.add_argument("--min-game-plies", type=int, default=None)
    parser.add_argument("--max-game-plies", type=int, default=None)
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
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional q-verifier checkpoint to initialize the shared encoder")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "q_probe")
    parser.add_argument(
        "--snapshot-every-batches",
        type=int,
        default=5000,
        help="Save an in-epoch snapshot every N batches; 0 disables",
    )
    parser.add_argument("--log-window", type=int, default=1000)
    parser.add_argument("--check-positive-weight", type=float, default=20.0, help="Initial class weight for positive CHECK labels in check loss")
    parser.add_argument("--check-positive-weight-end", type=float, default=1.0, help="Final positive CHECK class weight after decay")
    parser.add_argument("--check-positive-weight-decay-batches", type=int, default=150_000, help="Linearly decay CHECK positive class weight over this many total batches; 0 jumps to final weight")
    parser.add_argument("--mate-positive-weight", type=float, default=50.0, help="Class weight for positive MATE labels in mate loss")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="chess-gm")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-log-every", type=int, default=100)
    args = parser.parse_args()

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.bucket_plies < 1:
        raise ValueError("--bucket-plies must be >= 1")
    if args.snapshot_every_batches < 0:
        raise ValueError("--snapshot-every-batches must be >= 0")
    if args.check_positive_weight <= 0:
        raise ValueError("--check-positive-weight must be > 0")
    if args.check_positive_weight_end <= 0:
        raise ValueError("--check-positive-weight-end must be > 0")
    if args.check_positive_weight_decay_batches < 0:
        raise ValueError("--check-positive-weight-decay-batches must be >= 0")
    if args.mate_positive_weight <= 0:
        raise ValueError("--mate-positive-weight must be > 0")

    wandb_run = init_wandb(args)

    dataset = PlyProbeDataset(
        args.data_dir,
        context_plies=args.context_plies,
        max_probe_plies=args.max_probe_plies,
        min_game_plies=args.min_game_plies,
        max_game_plies=args.max_game_plies,
        examples_per_epoch=args.examples_per_epoch,
    )
    print(
        "dataset: "
        f"games={len(dataset.game_indices):,} examples_per_epoch={len(dataset):,} "
        f"context_plies={args.context_plies} max_probe_plies={args.max_probe_plies} "
        f"min_game_plies={args.min_game_plies} max_game_plies={args.max_game_plies}"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = QPlyProbeTransformer(
        vocab_size=len(VOCAB),
        ply_expr=8,
        model_dim=args.model_dim,
        heads=args.heads,
        history_layers=args.history_layers,
        q_layers=args.q_layers,
        num_queries=args.num_queries,
        dropout=args.dropout,
        pad_id=0,
    )
    if args.init_checkpoint is not None:
        load_encoder_checkpoint(model, args.init_checkpoint)
    model = model.to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mate_loss_weight = torch.tensor([1.0, args.mate_positive_weight], dtype=torch.float32, device=args.device)
    print(
        "loss weights: "
        f"check=[neg=1.0,pos={args.check_positive_weight:g}->{args.check_positive_weight_end:g} "
        f"over {args.check_positive_weight_decay_batches:,} batches] "
        f"mate=[neg=1.0,pos={args.mate_positive_weight:g}] "
        "turn=[neg=1.0,pos=1.0]"
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_batch = 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_window: deque[float] = deque(maxlen=args.log_window)
        check_loss_window: deque[float] = deque(maxlen=args.log_window)
        mate_loss_window: deque[float] = deque(maxlen=args.log_window)
        turn_loss_window: deque[float] = deque(maxlen=args.log_window)
        check_acc_window: deque[float] = deque(maxlen=args.log_window)
        mate_acc_window: deque[float] = deque(maxlen=args.log_window)
        turn_acc_window: deque[float] = deque(maxlen=args.log_window)
        prob_check_window: deque[float] = deque(maxlen=args.log_window)
        prob_mate_window: deque[float] = deque(maxlen=args.log_window)
        prob_black_window: deque[float] = deque(maxlen=args.log_window)
        check_tp_window: deque[int] = deque(maxlen=args.log_window)
        check_pos_window: deque[int] = deque(maxlen=args.log_window)
        check_pred_pos_window: deque[int] = deque(maxlen=args.log_window)
        mate_tp_window: deque[int] = deque(maxlen=args.log_window)
        mate_pos_window: deque[int] = deque(maxlen=args.log_window)
        mate_pred_pos_window: deque[int] = deque(maxlen=args.log_window)
        black_tp_window: deque[int] = deque(maxlen=args.log_window)
        black_pos_window: deque[int] = deque(maxlen=args.log_window)
        black_pred_pos_window: deque[int] = deque(maxlen=args.log_window)
        turn_correct_window: deque[int] = deque(maxlen=args.log_window)
        sample_count_window: deque[int] = deque(maxlen=args.log_window)
        bucket_prob_check_sum: dict[int, float] = {}
        bucket_check_correct: dict[int, int] = {}
        bucket_check_positive: dict[int, int] = {}
        bucket_check_pred_positive: dict[int, int] = {}
        bucket_check_tp: dict[int, int] = {}
        bucket_prob_mate_sum: dict[int, float] = {}
        bucket_mate_correct: dict[int, int] = {}
        bucket_mate_positive: dict[int, int] = {}
        bucket_mate_pred_positive: dict[int, int] = {}
        bucket_mate_tp: dict[int, int] = {}
        bucket_prob_black_sum: dict[int, float] = {}
        bucket_black_positive: dict[int, int] = {}
        bucket_black_pred_positive: dict[int, int] = {}
        bucket_black_tp: dict[int, int] = {}
        bucket_turn_correct: dict[int, int] = {}
        bucket_count: dict[int, int] = {}
        pbar = tqdm(loader, desc=f"q-probe epoch {epoch + 1}/{args.epochs}", unit="batch")
        for step, (x, check_y, mate_y, turn_y, probe_ply) in enumerate(pbar, start=1):
            global_batch += 1
            x = x.to(args.device, non_blocking=True)
            check_y = check_y.to(args.device, non_blocking=True)
            mate_y = mate_y.to(args.device, non_blocking=True)
            turn_y = turn_y.to(args.device, non_blocking=True)
            current_check_positive_weight = linear_decay(
                args.check_positive_weight,
                args.check_positive_weight_end,
                global_batch - 1,
                args.check_positive_weight_decay_batches,
            )
            check_loss_weight = torch.tensor([1.0, current_check_positive_weight], dtype=torch.float32, device=args.device)
            check_logits, mate_logits, turn_logits = model(x)
            check_loss = F.cross_entropy(check_logits, check_y, weight=check_loss_weight)
            mate_loss = F.cross_entropy(mate_logits, mate_y, weight=mate_loss_weight)
            turn_loss = F.cross_entropy(turn_logits, turn_y)
            loss = check_loss + mate_loss + turn_loss
            (loss / args.grad_accum_steps).backward()

            if step % args.grad_accum_steps == 0 or step == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            loss_window.append(float(loss.detach().cpu()))
            check_loss_window.append(float(check_loss.detach().cpu()))
            mate_loss_window.append(float(mate_loss.detach().cpu()))
            turn_loss_window.append(float(turn_loss.detach().cpu()))
            check_acc_window.append(accuracy(check_logits, check_y))
            mate_acc_window.append(accuracy(mate_logits, mate_y))
            turn_acc_window.append(accuracy(turn_logits, turn_y))
            prob_check = check_logits.softmax(dim=-1)[:, 1]
            prob_mate = mate_logits.softmax(dim=-1)[:, 1]
            prob_black = turn_logits.softmax(dim=-1)[:, 1]
            prob_check_window.append(float(prob_check.mean().detach().cpu()))
            prob_mate_window.append(float(prob_mate.mean().detach().cpu()))
            prob_black_window.append(float(prob_black.mean().detach().cpu()))
            check_pred = check_logits.argmax(dim=-1)
            mate_pred = mate_logits.argmax(dim=-1)
            turn_pred = turn_logits.argmax(dim=-1)
            check_tp_window.append(int(((check_pred == 1) & (check_y == 1)).sum().detach().cpu()))
            check_pos_window.append(int((check_y == 1).sum().detach().cpu()))
            check_pred_pos_window.append(int((check_pred == 1).sum().detach().cpu()))
            mate_tp_window.append(int(((mate_pred == 1) & (mate_y == 1)).sum().detach().cpu()))
            mate_pos_window.append(int((mate_y == 1).sum().detach().cpu()))
            mate_pred_pos_window.append(int((mate_pred == 1).sum().detach().cpu()))
            black_tp_window.append(int(((turn_pred == 1) & (turn_y == 1)).sum().detach().cpu()))
            black_pos_window.append(int((turn_y == 1).sum().detach().cpu()))
            black_pred_pos_window.append(int((turn_pred == 1).sum().detach().cpu()))
            turn_correct_window.append(int((turn_pred == turn_y).sum().detach().cpu()))
            sample_count_window.append(int(x.shape[0]))
            for (
                bucket,
                check_prob,
                check_correct,
                check_positive,
                check_pred_positive,
                check_true_positive,
                mate_prob,
                mate_correct,
                mate_positive,
                mate_pred_positive,
                mate_true_positive,
                black_prob,
                black_positive,
                black_pred_positive,
                black_true_positive,
                turn_correct,
            ) in zip(
                ((probe_ply - 1) // args.bucket_plies).tolist(),
                prob_check.detach().cpu().tolist(),
                (check_pred == check_y).detach().cpu().int().tolist(),
                check_y.detach().cpu().int().tolist(),
                (check_pred == 1).detach().cpu().int().tolist(),
                ((check_pred == 1) & (check_y == 1)).detach().cpu().int().tolist(),
                prob_mate.detach().cpu().tolist(),
                (mate_pred == mate_y).detach().cpu().int().tolist(),
                mate_y.detach().cpu().int().tolist(),
                (mate_pred == 1).detach().cpu().int().tolist(),
                ((mate_pred == 1) & (mate_y == 1)).detach().cpu().int().tolist(),
                prob_black.detach().cpu().tolist(),
                (turn_y == 1).detach().cpu().int().tolist(),
                (turn_pred == 1).detach().cpu().int().tolist(),
                ((turn_pred == 1) & (turn_y == 1)).detach().cpu().int().tolist(),
                (turn_pred == turn_y).detach().cpu().int().tolist(),
                strict=True,
            ):
                bucket = int(bucket)
                bucket_prob_check_sum[bucket] = bucket_prob_check_sum.get(bucket, 0.0) + float(check_prob)
                bucket_check_correct[bucket] = bucket_check_correct.get(bucket, 0) + int(check_correct)
                bucket_check_positive[bucket] = bucket_check_positive.get(bucket, 0) + int(check_positive)
                bucket_check_pred_positive[bucket] = bucket_check_pred_positive.get(bucket, 0) + int(check_pred_positive)
                bucket_check_tp[bucket] = bucket_check_tp.get(bucket, 0) + int(check_true_positive)
                bucket_prob_mate_sum[bucket] = bucket_prob_mate_sum.get(bucket, 0.0) + float(mate_prob)
                bucket_mate_correct[bucket] = bucket_mate_correct.get(bucket, 0) + int(mate_correct)
                bucket_mate_positive[bucket] = bucket_mate_positive.get(bucket, 0) + int(mate_positive)
                bucket_mate_pred_positive[bucket] = bucket_mate_pred_positive.get(bucket, 0) + int(mate_pred_positive)
                bucket_mate_tp[bucket] = bucket_mate_tp.get(bucket, 0) + int(mate_true_positive)
                bucket_prob_black_sum[bucket] = bucket_prob_black_sum.get(bucket, 0.0) + float(black_prob)
                bucket_black_positive[bucket] = bucket_black_positive.get(bucket, 0) + int(black_positive)
                bucket_black_pred_positive[bucket] = bucket_black_pred_positive.get(bucket, 0) + int(black_pred_positive)
                bucket_black_tp[bucket] = bucket_black_tp.get(bucket, 0) + int(black_true_positive)
                bucket_turn_correct[bucket] = bucket_turn_correct.get(bucket, 0) + int(turn_correct)
                bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
            rolling_loss = sum(loss_window) / len(loss_window)
            pbar.set_postfix(loss=rolling_loss)
            if wandb_run is not None and (step % args.wandb_log_every == 0 or step == len(loader)):
                check_pos = sum(check_pos_window)
                mate_pos = sum(mate_pos_window)
                check_pred_pos = sum(check_pred_pos_window)
                mate_pred_pos = sum(mate_pred_pos_window)
                black_pos = sum(black_pos_window)
                black_pred_pos = sum(black_pred_pos_window)
                black_tp = sum(black_tp_window)
                log_payload = {
                    "train/loss": rolling_loss,
                    "train/check_loss": sum(check_loss_window) / len(check_loss_window),
                    "train/mate_loss": sum(mate_loss_window) / len(mate_loss_window),
                    "train/turn_loss": sum(turn_loss_window) / len(turn_loss_window),
                    "train/check_acc": sum(check_acc_window) / len(check_acc_window),
                    "train/mate_acc": sum(mate_acc_window) / len(mate_acc_window),
                    "train/turn_acc": sum(turn_acc_window) / len(turn_acc_window),
                    "train/prob_check": sum(prob_check_window) / len(prob_check_window),
                    "train/prob_mate": sum(prob_mate_window) / len(prob_mate_window),
                    "train/prob_black": sum(prob_black_window) / len(prob_black_window),
                    "train/p_check": sum(check_tp_window) / check_pred_pos if check_pred_pos else 0.0,
                    "train/r_check": sum(check_tp_window) / check_pos if check_pos else 0.0,
                    "train/p_mate": sum(mate_tp_window) / mate_pred_pos if mate_pred_pos else 0.0,
                    "train/r_mate": sum(mate_tp_window) / mate_pos if mate_pos else 0.0,
                    "train/p_black": black_tp / black_pred_pos if black_pred_pos else 0.0,
                    "train/r_black": black_tp / black_pos if black_pos else 0.0,
                    "train/check_positive": check_pos,
                    "train/check_pred_positive": check_pred_pos,
                    "train/check_tp": sum(check_tp_window),
                    "train/mate_positive": mate_pos,
                    "train/mate_pred_positive": mate_pred_pos,
                    "train/mate_tp": sum(mate_tp_window),
                    "train/black_positive": black_pos,
                    "train/black_pred_positive": black_pred_pos,
                    "train/black_tp": black_tp,
                    "train/turn_correct": sum(turn_correct_window),
                    "train/n": sum(sample_count_window),
                    "loss_weight/check_positive": current_check_positive_weight,
                    "loss_weight/check_positive_start": args.check_positive_weight,
                    "loss_weight/check_positive_end": args.check_positive_weight_end,
                    "loss_weight/mate_positive": args.mate_positive_weight,
                    "epoch": epoch + 1,
                    "batch": step,
                }
                for bucket in sorted(bucket_count):
                    lo = bucket * args.bucket_plies + 1
                    hi = (bucket + 1) * args.bucket_plies
                    prefix = f"bucket/{lo}_{hi}"
                    log_payload[f"{prefix}/prob_check"] = bucket_prob_check_sum[bucket] / bucket_count[bucket]
                    log_payload[f"{prefix}/p_check"] = (
                        bucket_check_tp[bucket] / bucket_check_pred_positive[bucket]
                        if bucket_check_pred_positive[bucket]
                        else 0.0
                    )
                    log_payload[f"{prefix}/r_check"] = (
                        bucket_check_tp[bucket] / bucket_check_positive[bucket]
                        if bucket_check_positive[bucket]
                        else 0.0
                    )
                    log_payload[f"{prefix}/check_positive"] = bucket_check_positive[bucket]
                    log_payload[f"{prefix}/check_pred_positive"] = bucket_check_pred_positive[bucket]
                    log_payload[f"{prefix}/check_tp"] = bucket_check_tp[bucket]
                    log_payload[f"{prefix}/prob_mate"] = bucket_prob_mate_sum[bucket] / bucket_count[bucket]
                    log_payload[f"{prefix}/p_mate"] = (
                        bucket_mate_tp[bucket] / bucket_mate_pred_positive[bucket]
                        if bucket_mate_pred_positive[bucket]
                        else 0.0
                    )
                    log_payload[f"{prefix}/r_mate"] = (
                        bucket_mate_tp[bucket] / bucket_mate_positive[bucket]
                        if bucket_mate_positive[bucket]
                        else 0.0
                    )
                    log_payload[f"{prefix}/mate_positive"] = bucket_mate_positive[bucket]
                    log_payload[f"{prefix}/mate_pred_positive"] = bucket_mate_pred_positive[bucket]
                    log_payload[f"{prefix}/mate_tp"] = bucket_mate_tp[bucket]
                    log_payload[f"{prefix}/prob_black"] = bucket_prob_black_sum[bucket] / bucket_count[bucket]
                    log_payload[f"{prefix}/p_black"] = (
                        bucket_black_tp[bucket] / bucket_black_pred_positive[bucket]
                        if bucket_black_pred_positive[bucket]
                        else 0.0
                    )
                    log_payload[f"{prefix}/r_black"] = (
                        bucket_black_tp[bucket] / bucket_black_positive[bucket]
                        if bucket_black_positive[bucket]
                        else 0.0
                    )
                    log_payload[f"{prefix}/black_positive"] = bucket_black_positive[bucket]
                    log_payload[f"{prefix}/black_pred_positive"] = bucket_black_pred_positive[bucket]
                    log_payload[f"{prefix}/black_tp"] = bucket_black_tp[bucket]
                    log_payload[f"{prefix}/turn_acc"] = bucket_turn_correct[bucket] / bucket_count[bucket]
                    log_payload[f"{prefix}/n"] = bucket_count[bucket]
                    log_payload[f"{prefix}/turn_correct"] = bucket_turn_correct[bucket]
                wandb_run.log(log_payload, step=(epoch * len(loader)) + step)

            if step % 100 == 0 or step == len(loader):
                check_pos = sum(check_pos_window)
                check_pred_pos = sum(check_pred_pos_window)
                mate_pos = sum(mate_pos_window)
                mate_pred_pos = sum(mate_pred_pos_window)
                black_pos = sum(black_pos_window)
                black_pred_pos = sum(black_pred_pos_window)
                black_tp = sum(black_tp_window)
                metric_table = format_table(
                    ["metric", "value"],
                    [
                        ["loss", f"{rolling_loss:.4f}"],
                        ["check_loss", f"{sum(check_loss_window) / len(check_loss_window):.4f}"],
                        ["mate_loss", f"{sum(mate_loss_window) / len(mate_loss_window):.4f}"],
                        ["turn_loss", f"{sum(turn_loss_window) / len(turn_loss_window):.4f}"],
                        ["check_acc", f"{sum(check_acc_window) / len(check_acc_window):.4f}"],
                        ["mate_acc", f"{sum(mate_acc_window) / len(mate_acc_window):.4f}"],
                        ["turn_acc", f"{sum(turn_acc_window) / len(turn_acc_window):.4f}"],
                        ["prob_check", f"{sum(prob_check_window) / len(prob_check_window):.4f}"],
                        ["prob_mate", f"{sum(prob_mate_window) / len(prob_mate_window):.4f}"],
                        ["prob_black", f"{sum(prob_black_window) / len(prob_black_window):.4f}"],
                        ["p_check", f"{(sum(check_tp_window) / check_pred_pos if check_pred_pos else 0.0):.4f}"],
                        ["r_check", f"{(sum(check_tp_window) / check_pos if check_pos else 0.0):.4f}"],
                        ["p_mate", f"{(sum(mate_tp_window) / mate_pred_pos if mate_pred_pos else 0.0):.4f}"],
                        ["r_mate", f"{(sum(mate_tp_window) / mate_pos if mate_pos else 0.0):.4f}"],
                        ["p_black", f"{(black_tp / black_pred_pos if black_pred_pos else 0.0):.4f}"],
                        ["r_black", f"{(black_tp / black_pos if black_pos else 0.0):.4f}"],
                    ],
                )
                bucket_rows = []
                for bucket in sorted(bucket_count):
                    bucket_rows.append(
                        [
                            f"{bucket * args.bucket_plies + 1}-{(bucket + 1) * args.bucket_plies}",
                            f"{(bucket_check_tp[bucket] / bucket_check_pred_positive[bucket] if bucket_check_pred_positive[bucket] else 0.0):.3f}",
                            f"{(bucket_check_tp[bucket] / bucket_check_positive[bucket] if bucket_check_positive[bucket] else 0.0):.3f}",
                            str(bucket_check_positive[bucket]),
                            str(bucket_check_tp[bucket]),
                            f"{(bucket_mate_tp[bucket] / bucket_mate_pred_positive[bucket] if bucket_mate_pred_positive[bucket] else 0.0):.3f}",
                            f"{(bucket_mate_tp[bucket] / bucket_mate_positive[bucket] if bucket_mate_positive[bucket] else 0.0):.3f}",
                            str(bucket_mate_positive[bucket]),
                            str(bucket_mate_tp[bucket]),
                            f"{(bucket_black_tp[bucket] / bucket_black_pred_positive[bucket] if bucket_black_pred_positive[bucket] else 0.0):.3f}",
                            f"{(bucket_black_tp[bucket] / bucket_black_positive[bucket] if bucket_black_positive[bucket] else 0.0):.3f}",
                            str(bucket_black_positive[bucket]),
                            str(bucket_black_pred_positive[bucket]),
                            str(bucket_black_tp[bucket]),
                            f"{bucket_turn_correct[bucket] / bucket_count[bucket]:.3f}",
                            str(bucket_count[bucket]),
                        ]
                    )
                bucket_table = format_table(
                    [
                        "plies",
                        "p_chk",
                        "r_chk",
                        "chk+",
                        "chk_tp",
                        "p_mate",
                        "r_mate",
                        "mate+",
                        "mate_tp",
                        "p_blk",
                        "r_blk",
                        "blk+",
                        "blk_pred",
                        "blk_tp",
                        "turn_a",
                        "n",
                    ],
                    bucket_rows,
                )
                pbar.write(
                    f"\nepoch={epoch + 1} batch={step}/{len(loader)}\n"
                    f"\nsummary\n{metric_table}\n"
                    f"\nprobe buckets\n{bucket_table}"
                )
                bucket_prob_check_sum.clear()
                bucket_check_correct.clear()
                bucket_check_positive.clear()
                bucket_check_pred_positive.clear()
                bucket_check_tp.clear()
                bucket_prob_mate_sum.clear()
                bucket_mate_correct.clear()
                bucket_mate_positive.clear()
                bucket_mate_pred_positive.clear()
                bucket_mate_tp.clear()
                bucket_prob_black_sum.clear()
                bucket_black_positive.clear()
                bucket_black_pred_positive.clear()
                bucket_black_tp.clear()
                bucket_turn_correct.clear()
                bucket_count.clear()

            if args.snapshot_every_batches and step % args.snapshot_every_batches == 0:
                snapshot_path = args.checkpoint_dir / f"q_probe_epoch_{epoch + 1:03d}_batch_{step:06d}.pt"
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

        ckpt_path = args.checkpoint_dir / f"q_probe_epoch_{epoch + 1}.pt"
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
