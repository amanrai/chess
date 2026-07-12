"""Q-Former style verifier network.

Variable-length ply history -> fixed [B, K, D] query output -> verifier logits.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from chessgm.network import PlyHistoryEncoder


class CrossAttentionBlock(nn.Module):
    """Query tokens cross-attend to encoded ply-history tokens."""

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


class QFormerPlyHistoryEncoder(nn.Module):
    """Encode variable ply history into fixed learned query slots.

    Input:
      x_ids: [B, T, S]

    Output:
      q: [B, K, D]
    """

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
        self.ply_expr = ply_expr
        self.model_dim = model_dim
        self.num_queries = num_queries
        self.pad_id = pad_id
        self.history_encoder = PlyHistoryEncoder(
            vocab_size=vocab_size,
            ply_expr=ply_expr,
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


class QueryTransformerBlock(nn.Module):
    """Self-attention refinement over the unpooled Q-Former query bank."""

    def __init__(self, model_dim: int, heads: int, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_attn = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_attn(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.mlp(self.ln_mlp(x))


def load_pretrained_qformer_encoder(
    encoder: QFormerPlyHistoryEncoder,
    checkpoint_path: str | Path,
    *,
    freeze: bool = False,
) -> int:
    """Load only Q-Former encoder weights from a probe or verifier checkpoint.

    Probe checkpoints prefix the reusable weights with ``encoder.``; classification
    heads are deliberately ignored. Returns the number of loaded tensors.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model_state = encoder.state_dict()
    compatible = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
        and key.removeprefix("encoder.") in model_state
        and model_state[key.removeprefix("encoder.")].shape == value.shape
    }
    if not compatible:
        raise ValueError(f"no compatible Q-Former encoder weights found in {checkpoint_path}")
    encoder.load_state_dict(compatible, strict=False)
    if freeze:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    return len(compatible)


class DiffThinkerMLP(nn.Module):
    """Single-query cross-attention readout over an unpooled Q-Former state bank.

    A learned query asks one task-specific question of all Q-Former slots. Its
    single-head attention result is refined by an MLP and returned as [B, D],
    ready for a task head without mean-pooling the state bank.
    """

    def __init__(self, model_dim: int, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.readout_query = nn.Parameter(torch.randn(1, model_dim) * 0.02)
        self.ln_query = nn.LayerNorm(model_dim)
        self.ln_context = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=1,
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
        self.ln_f = nn.LayerNorm(model_dim)

    def forward(self, q_bank: torch.Tensor) -> torch.Tensor:
        """Read a single state vector from a Q-Former bank [B, K, D]."""
        if q_bank.ndim != 3:
            raise ValueError(f"expected Q-bank [B, K, D], got {tuple(q_bank.shape)}")
        queries = self.readout_query[None, :, :].expand(q_bank.shape[0], -1, -1)
        context = self.ln_context(q_bank)
        attended, _ = self.cross_attn(
            query=self.ln_query(queries),
            key=context,
            value=context,
            need_weights=False,
        )
        readout = queries + attended
        readout = readout + self.mlp(self.ln_mlp(readout))
        return self.ln_f(readout).squeeze(1)


class QInverseTransitionDecoder(nn.Module):
    """Decode an immediate ply from Q-Former states before and after that ply.

    A single shared Q-Former encodes both histories. The after-state query bank
    cross-attends to the before-state bank (Q=after, K/V=before), preserving all
    learned Q slots through transition refinement and move-slot decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        ply_expr: int = 8,
        model_dim: int = 256,
        heads: int = 8,
        history_layers: int = 4,
        q_layers: int = 2,
        num_queries: int = 16,
        transition_layers: int = 2,
        dropout: float = 0.0,
        pad_id: int = 0,
        pretrained_encoder_checkpoint: str | Path | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        if transition_layers < 0:
            raise ValueError("transition_layers must be >= 0")
        self.ply_expr = ply_expr
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
        if pretrained_encoder_checkpoint is not None:
            self.loaded_encoder_tensors = load_pretrained_qformer_encoder(
                self.encoder, pretrained_encoder_checkpoint, freeze=freeze_encoder
            )
        else:
            self.loaded_encoder_tensors = 0
            if freeze_encoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad_(False)

        # Q is the desired successor state; K/V are the preceding state.
        self.inverse_attention = CrossAttentionBlock(model_dim, heads, dropout=dropout)
        # Direct state highways retain per-query information beyond cross-attention.
        self.after_highway = nn.Linear(model_dim, model_dim, bias=False)
        self.before_highway = nn.Linear(model_dim, model_dim, bias=False)
        self.transition_blocks = nn.ModuleList(
            [
                QueryTransformerBlock(model_dim, heads, dropout=dropout)
                for _ in range(transition_layers)
            ]
        )
        self.slot_queries = nn.Parameter(torch.randn(ply_expr, model_dim) * 0.02)
        self.slot_attention = CrossAttentionBlock(model_dim, heads, dropout=dropout)
        self.slot_ln = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocab_size)

    def forward(self, before_ids: torch.Tensor, after_ids: torch.Tensor) -> torch.Tensor:
        """Predict the ply packet mapping ``before_ids`` to ``after_ids``.

        Args:
            before_ids: [B, T, S] history through state t.
            after_ids: [B, T, S] history through state t+1.

        Returns:
            Logits [B, S, vocab_size], one distribution per move-packet slot.
        """
        if before_ids.shape != after_ids.shape:
            raise ValueError(
                f"before_ids and after_ids must have matching shapes, got "
                f"{tuple(before_ids.shape)} and {tuple(after_ids.shape)}"
            )
        if before_ids.ndim != 3 or before_ids.shape[-1] != self.ply_expr:
            raise ValueError(f"expected [B, T, {self.ply_expr}] state histories")

        z_before = self.encoder(before_ids)
        z_after = self.encoder(after_ids)
        transition = self.inverse_attention(z_after, z_before, key_padding_mask=None)
        transition = transition + self.after_highway(z_after) + self.before_highway(z_before)
        for block in self.transition_blocks:
            transition = block(transition)

        slots = self.slot_queries[None, :, :].expand(before_ids.shape[0], -1, -1)
        slots = self.slot_attention(slots, transition, key_padding_mask=None)
        return self.head(self.slot_ln(slots))


class QVerifierTransformer(nn.Module):
    """Verifier using Q-Former fixed-size encoder output."""

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
