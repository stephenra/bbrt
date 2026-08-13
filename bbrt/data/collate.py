"""Batch padding shared by the datamodules and the decoder."""

from __future__ import annotations

import torch


def pad_sequences(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    """Right-pad id sequences into a ``(N, max_len)`` long tensor.

    Callers derive the pad mask with ``out.eq(pad_id)``.
    """
    maxlen = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out
