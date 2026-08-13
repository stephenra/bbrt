"""From-scratch Transformer encoder-decoder for SELFIES translation.

Pre-norm blocks, RMSNorm, RoPE on self-attention, SwiGLU FFN, and weight-tied
shared source/target embeddings + output head (matching the original repo's
``-share_embeddings -share_vocab``). The encoder stack is the reusable
:class:`bbrt.models.encoder.TransformerEncoder`, also used by the DRD2 classifier.
"""

from __future__ import annotations

import torch
from torch import nn

from bbrt.config import ModelConfig
from bbrt.models.components import (
    MultiHeadAttention,
    RMSNorm,
    SwiGLU,
    init_transformer_weights,
)
from bbrt.models.encoder import TransformerEncoder


class DecoderLayer(nn.Module):
    """Pre-norm masked self-attention + cross-attention + SwiGLU block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm3 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        tgt_pad_mask: torch.Tensor | None,
        mem_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(
            self.self_attn(h, h, cos, sin, key_padding_mask=tgt_pad_mask, is_causal=True)
        )
        h = self.norm2(x)
        # Cross-attention: no RoPE (memory already carries positional info).
        x = x + self.dropout(self.cross_attn(h, memory, key_padding_mask=mem_pad_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

    def step(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mem_pad_mask: torch.Tensor | None,
        cache: dict,
    ) -> torch.Tensor:
        """One incremental decode step, reading/writing a per-layer KV cache.

        ``cache["self"]`` grows by one position per call; ``cache["cross"]`` is
        computed once from ``memory`` and reused thereafter.
        """
        h = self.norm1(x)
        a, cache["self"] = self.self_attn(
            h,
            h,
            cos,
            sin,
            key_padding_mask=None,
            is_causal=False,
            past=cache.get("self"),
            append=True,
            return_kv=True,
        )
        x = x + self.dropout(a)
        h = self.norm2(x)
        if "cross" not in cache:
            c, cache["cross"] = self.cross_attn(
                h,
                memory,
                key_padding_mask=mem_pad_mask,
                past=None,
                append=False,
                return_kv=True,
            )
        else:
            c = self.cross_attn(
                h,
                memory,
                key_padding_mask=mem_pad_mask,
                past=cache["cross"],
                append=False,
            )
        x = x + self.dropout(c)
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class TransformerSeq2Seq(nn.Module):
    """Encoder-decoder Transformer over a shared SELFIES vocabulary."""

    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.vocab_size = vocab_size

        self.enc = TransformerEncoder(cfg, vocab_size, pad_id=pad_id)
        self.dropout = nn.Dropout(cfg.dropout)
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.dec_norm = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.enc.embed.weight

        self.apply(init_transformer_weights)

    @property
    def embed(self) -> nn.Embedding:
        """The shared token embedding (source/target/output are tied to it)."""
        return self.enc.embed

    # -- encoder ------------------------------------------------------------ #
    def encode(self, src: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        return self.enc(src, src_pad_mask)

    # -- decoder ------------------------------------------------------------ #
    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad_mask: torch.Tensor | None,
        mem_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cos, sin = self.enc.rope(tgt_in.size(1))
        cos, sin = cos.to(tgt_in.device), sin.to(tgt_in.device)
        x = self.dropout(self.enc.embed(tgt_in) * self.enc.embed_scale)
        for layer in self.decoder:
            x = layer(x, memory, cos, sin, tgt_pad_mask, mem_pad_mask)
        x = self.dec_norm(x)
        return self.lm_head(x)

    # -- incremental (KV-cached) decoding ---------------------------------- #
    def init_caches(self) -> list[dict]:
        """Fresh empty per-decoder-layer KV caches for one decode run."""
        return [{} for _ in self.decoder]

    def decode_step(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        mem_pad_mask: torch.Tensor | None,
        caches: list[dict],
        position: int,
    ) -> torch.Tensor:
        """Decode a single position with KV caching.

        ``token_ids`` is ``(B, 1)`` (the token at absolute ``position``);
        returns logits ``(B, 1, vocab)``. Mathematically identical to a full
        :meth:`decode` over the same prefix, but O(1) per step instead of O(L).
        """
        cos, sin = self.enc.rope.at(position, token_ids.size(1))
        cos, sin = cos.to(token_ids.device), sin.to(token_ids.device)
        x = self.dropout(self.enc.embed(token_ids) * self.enc.embed_scale)
        for layer, cache in zip(self.decoder, caches, strict=True):
            x = layer.step(x, memory, cos, sin, mem_pad_mask, cache)  # type: ignore[operator]
        x = self.dec_norm(x)
        return self.lm_head(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_in_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward. Returns logits ``(B, T_tgt, vocab)``."""
        memory = self.encode(src, src_pad_mask)
        return self.decode(tgt_in, memory, tgt_in_pad_mask, src_pad_mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
