"""Modern Transformer building blocks: RMSNorm, RoPE, MHA, SwiGLU FFN."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (Su et al., 2021).

    Produces (cos, sin) tables for a given sequence length; applied to query
    and key projections inside self-attention.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.at(0, seq_len)

    def at(self, start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) rows for absolute positions ``[start, start+length)``.

        Used for incremental (KV-cached) decoding, where the query/key at each
        step sits at a single absolute position.
        """
        end = start + length
        if end > self.max_seq_len:
            raise ValueError(f"position {end} exceeds RoPE table {self.max_seq_len}")
        return self.cos[start:end], self.sin[start:end]  # type: ignore[index]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape (B, H, T, Dh). cos/sin are (T, Dh)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    """Multi-head attention using fused scaled-dot-product attention.

    Handles both self-attention (with optional RoPE + causal masking) and
    cross-attention (no RoPE). Key padding is supplied as a boolean mask where
    ``True`` marks padded positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        append: bool = False,
        return_kv: bool = False,
    ):
        """Attention with optional KV cache.

        ``past``/``append``/``return_kv`` drive incremental decoding:

        * self-attention step: ``append=True`` concatenates the new key/value
          onto ``past`` (RoPE is applied to the new key at its position);
        * cross-attention step: ``append=False`` with ``past`` set reuses the
          cached memory key/value (skipping the projections entirely).

        With all three at their defaults this is the standard full-sequence
        attention used during training.
        """
        b, tq, _ = x_q.shape
        q = self._shape(self.q_proj(x_q))  # (B, H, Tq, Dh)

        if past is not None and not append:
            # Cross-attention: reuse cached memory projections as-is.
            k, v = past
        else:
            k = self._shape(self.k_proj(x_kv))
            v = self._shape(self.v_proj(x_kv))
            if cos is not None and sin is not None:
                k = apply_rope(k, cos, sin)
            if append and past is not None:
                pk, pv = past
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)

        tk = k.size(2)
        attn_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, Tk) additive float mask; -inf on padded keys.
            attn_mask = torch.zeros(b, 1, 1, tk, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            if is_causal:
                causal = torch.triu(
                    torch.ones(tq, tk, dtype=torch.bool, device=q.device), diagonal=1
                )
                attn_mask = attn_mask.masked_fill(causal[None, None, :, :], float("-inf"))
            is_causal = False  # folded into attn_mask

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, tq, -1)
        out = self.out_proj(out)
        if return_kv:
            return out, (k, v)
        return out


def init_transformer_weights(module: nn.Module) -> None:
    """Standard small-init for Transformer submodules.

    Linear/Embedding weights ~ N(0, 0.02); biases zeroed; the embedding's
    padding row zeroed. Safe to ``model.apply(...)`` over the whole model.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()  # type: ignore[index]


class AttentionPool(nn.Module):
    """Learned-query attention pooling over token states (mask-aware).

    A single learned query attends over the encoder outputs, producing one
    vector per sequence. Typically a better classification summary than a plain
    masked mean because it can weight informative tokens.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.query, std=0.02)
        self.scale = d_model**-0.5

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        scores = (h @ self.query) * self.scale  # (B, T)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (attn * h).sum(dim=1)  # (B, D)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        hidden = int(2 * d_ff / 3)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))
