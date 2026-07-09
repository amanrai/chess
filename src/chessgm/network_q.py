"""Q-Former style verifier network.

Variable-length move history -> fixed [B, K, D] query output -> verifier logits.
"""
from __future__ import annotations

import torch
from torch import nn

from chessgm.network import MoveHistoryEncoder


class CrossAttentionBlock(nn.Module):
    """Query tokens cross-attend to encoded move/history tokens."""

    def __init__(self, model_dim: int, heads: int, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(model_dim)
        self.ln_ctx = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_mlp = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_mult * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.ln_q(queries)
        ctx = self.ln_ctx(context)
        attended, _ = self.cross_attn(
            query=q,
            key=ctx,
            value=ctx,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attended
        queries = queries + self.mlp(self.ln_mlp(queries))
        return queries


class QFormerMoveHistoryEncoder(nn.Module):
    """Encode variable move history into fixed learned query slots.

    Input:
      x_ids: [B, T, S]

    Output:
      q: [B, K, D]
    """

    def __init__(
        self,
        vocab_size: int,
        move_expr: int = 8,
        model_dim: int = 256,
        heads: int = 8,
        history_layers: int = 4,
        q_layers: int = 2,
        num_queries: int = 16,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.move_expr = move_expr
        self.model_dim = model_dim
        self.num_queries = num_queries
        self.pad_id = pad_id
        self.history_encoder = MoveHistoryEncoder(
            vocab_size=vocab_size,
            move_expr=move_expr,
            model_dim=model_dim,
            heads=heads,
            layers=history_layers,
            dropout=dropout,
            pad_id=pad_id,
            causal=True,
        )
        self.query_tokens = nn.Parameter(torch.randn(num_queries, model_dim) * 0.02)
        self.q_blocks = nn.ModuleList(
            [CrossAttentionBlock(model_dim, heads, dropout=dropout) for _ in range(q_layers)]
        )
        self.ln_f = nn.LayerNorm(model_dim)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        b, t, s = x_ids.shape
        context = self.history_encoder(x_ids)  # [B, T*S, D]
        key_padding_mask = x_ids.reshape(b, t * s).eq(self.pad_id)  # True means ignore key
        queries = self.query_tokens[None, :, :].expand(b, -1, -1)
        for block in self.q_blocks:
            queries = block(queries, context, key_padding_mask=key_padding_mask)
        return self.ln_f(queries)


class QVerifierTransformer(nn.Module):
    """Verifier using Q-Former fixed-size encoder output."""

    def __init__(
        self,
        vocab_size: int,
        move_expr: int = 8,
        model_dim: int = 256,
        heads: int = 8,
        history_layers: int = 4,
        q_layers: int = 2,
        num_queries: int = 16,
        dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.encoder = QFormerMoveHistoryEncoder(
            vocab_size=vocab_size,
            move_expr=move_expr,
            model_dim=model_dim,
            heads=heads,
            history_layers=history_layers,
            q_layers=q_layers,
            num_queries=num_queries,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 3),
        )

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        q = self.encoder(x_ids)  # [B, K, D]
        pooled = q.mean(dim=1)
        return self.classifier(pooled)
