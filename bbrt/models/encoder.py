"""Reusable Transformer encoder.

Shared by the seq2seq model (:mod:`bbrt.models.transformer`) and the DRD2
property classifier (:mod:`bbrt.models.drd2_model`), so the two never drift.
Owns the token embedding, RoPE table, pre-norm encoder-layer stack, and a final
RMSNorm; exposes a mask-aware mean-pool for downstream classification heads.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from bbrt.config import ModelConfig
from bbrt.models.components import MultiHeadAttention, RMSNorm, RotaryEmbedding, SwiGLU


class EncoderLayer(nn.Module):
    """Pre-norm self-attention + SwiGLU block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        src_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, cos, sin, key_padding_mask=src_pad_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Token embedding + RoPE + pre-norm encoder stack.

    ``forward`` returns per-token hidden states ``(B, T, d_model)``; ``pooled``
    returns a single mask-aware mean-pooled vector ``(B, d_model)`` per sequence.
    Weight initialization is left to the owning module (call
    ``self.apply(init_transformer_weights)`` there).
    """

    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.embed_scale = math.sqrt(cfg.d_model)
        self.rope = RotaryEmbedding(
            cfg.d_model // cfg.n_heads, cfg.max_seq_len, theta=cfg.rope_theta
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Encode token ids to hidden states. ``pad_mask``: True marks padding."""
        cos, sin = self.rope(ids.size(1))
        cos, sin = cos.to(ids.device), sin.to(ids.device)
        x = self.dropout(self.embed(ids) * self.embed_scale)
        for layer in self.layers:
            x = layer(x, cos, sin, pad_mask)
        return self.norm(x)

    def pooled(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Mask-aware mean pool over non-pad positions -> ``(B, d_model)``."""
        h = self.forward(ids, pad_mask)
        keep = (~pad_mask).unsqueeze(-1).type_as(h)  # (B, T, 1), 1.0 on real tokens
        summed = (h * keep).sum(dim=1)
        counts = keep.sum(dim=1).clamp_min(1.0)
        return summed / counts
