"""Neural network modules for chess move-packet models."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ArtisanalEmbedder(nn.Module):
    """Token embedding plus optional within-move slot embedding.

    Input shape:
      ids: [B, T, S]

    Output shape:
      x: [B, T, S, D]

    `S` is the number of tokens used to express a single move, currently 8.
    """

    def __init__(self, vocab_size: int, model_dim: int, within_move_positions: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.within_move_positions = within_move_positions
        self.move_pos_emb = (
            nn.Embedding(within_move_positions, model_dim) if within_move_positions else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x)
        if self.move_pos_emb is not None:
            if x.shape[-1] != self.within_move_positions:
                raise ValueError(
                    f"expected last dim {self.within_move_positions}, got {x.shape[-1]}"
                )
            p = torch.arange(self.within_move_positions, device=x.device)
            out = out + self.move_pos_emb(p)
        return out


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, move_pos: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to q/k.

    Args:
      x: [B, H, N, Dh]
      move_pos: [N], position id for each flattened token.
        For chess move packets this repeats the move index for all slots:
        [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, ...]
    """
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got {dim}")
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    freqs = torch.einsum("n,d->nd", move_pos.float(), inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


def chess_move_positions(num_moves: int, move_expr: int, device: torch.device) -> torch.Tensor:
    """Build move-level positions for flattened move packets."""
    return torch.arange(num_moves, device=device).repeat_interleave(move_expr)


class ArtisanalRoPEAttention(nn.Module):
    """Multi-head self-attention with chess move-level RoPE."""

    def __init__(self, model_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by heads={heads}")
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")

        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        move_pos: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention.

        Args:
          x: [B, N, D]
          move_pos: [N]
          attn_mask: optional bool mask broadcastable as [B or 1, H or 1, N, N].
            True means allowed, False means blocked.
        """
        b, n, d = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        qkv = qkv.view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, N, H, Dh]

        q = q.transpose(1, 2)  # [B, H, N, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, move_pos)
        k = apply_rope(k, move_pos)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]
        query_has_valid_key = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]
            query_has_valid_key = attn_mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~attn_mask, float("-inf"))
            # Fully masked query rows otherwise become softmax(-inf, ...) = NaN.
            scores = scores.masked_fill(~query_has_valid_key, 0.0)

        attn = self.dropout(scores.softmax(dim=-1))
        if query_has_valid_key is not None:
            attn = attn.masked_fill(~query_has_valid_key, 0.0)
        y = attn @ v  # [B, H, N, Dh]
        y = y.transpose(1, 2).contiguous().view(b, n, d)
        return self.out(y)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(
        self,
        model_dim: int,
        heads: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = ArtisanalRoPEAttention(model_dim, heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_mult * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        move_pos: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), move_pos, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


def build_move_causal_mask(num_moves: int, move_expr: int, device: torch.device) -> torch.Tensor:
    """Allow attention to current/prior moves, never future moves."""
    move_idx = chess_move_positions(num_moves, move_expr, device)
    return move_idx[:, None] >= move_idx[None, :]


def build_key_padding_attention_mask(x_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Build [B, N, N] mask that prevents attention to padded key tokens.

    Queries are left alone; keys that are PAD are blocked.
    """
    b, t, s = x_ids.shape
    key_ok = (x_ids.reshape(b, t * s) != pad_id)
    return key_ok[:, None, :].expand(b, t * s, t * s)


class MoveHistoryEncoder(nn.Module):
    """Encode variable move history tokens with move-level RoPE."""

    def __init__(
        self,
        vocab_size: int,
        move_expr: int = 8,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.0,
        pad_id: int = 0,
        causal: bool = True,
    ):
        super().__init__()
        self.move_expr = move_expr
        self.model_dim = model_dim
        self.pad_id = pad_id
        self.causal = causal
        self.embedder = ArtisanalEmbedder(vocab_size, model_dim, within_move_positions=move_expr)
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_dim, heads, dropout=dropout) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(model_dim)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """Encode move packets.

        Args:
          x_ids: [B, T, S]

        Returns:
          encoded: [B, T*S, D]
        """
        b, t, s = x_ids.shape
        if s != self.move_expr:
            raise ValueError(f"expected move_expr={self.move_expr}, got {s}")

        x = self.embedder(x_ids)  # [B, T, S, D]
        x = x.reshape(b, t * s, self.model_dim)
        move_pos = chess_move_positions(t, s, x.device)

        attn_mask = build_key_padding_attention_mask(x_ids, self.pad_id)
        if self.causal:
            attn_mask = attn_mask & build_move_causal_mask(t, s, x.device)[None, :, :]

        for block in self.blocks:
            x = block(x, move_pos, attn_mask)
        return self.ln_f(x)


class VerifierTransformer(nn.Module):
    """Prefix result classifier.

    Input:
      x_ids: [B, T, 8]

    Output:
      logits: [B, 3] for white win / black win / draw.
    """

    def __init__(
        self,
        vocab_size: int,
        move_expr: int = 8,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.move_expr = move_expr
        self.encoder = MoveHistoryEncoder(
            vocab_size=vocab_size,
            move_expr=move_expr,
            model_dim=model_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            pad_id=pad_id,
            causal=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 3),
        )

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        b, t, s = x_ids.shape
        encoded = self.encoder(x_ids)  # [B, T*S, D]
        final_move = encoded[:, -s:, :].mean(dim=1)
        return self.classifier(final_move)


class PacketARGenerator(nn.Module):
    """Simple next-move packet generator baseline.

    Consumes a move-history prefix and predicts all slots of the next packet jointly.
    """

    def __init__(
        self,
        vocab_size: int,
        move_expr: int = 8,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.move_expr = move_expr
        self.encoder = MoveHistoryEncoder(
            vocab_size=vocab_size,
            move_expr=move_expr,
            model_dim=model_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            pad_id=pad_id,
            causal=True,
        )
        self.next_slot_queries = nn.Parameter(torch.randn(move_expr, model_dim) / math.sqrt(model_dim))
        self.head = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        b, _t, s = x_ids.shape
        encoded = self.encoder(x_ids)
        context = encoded[:, -s:, :].mean(dim=1)
        slot_states = context[:, None, :] + self.next_slot_queries[None, :, :]
        return self.head(slot_states)  # [B, S, vocab]
