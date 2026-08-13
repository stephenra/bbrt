"""LightningDataModule over the tokenized-SELFIES parallel corpus."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from bbrt.config import DataConfig
from bbrt.data.collate import pad_sequences
from bbrt.data.tokenizer import SelfiesTokenizer


def _read_lines(path: str | Path) -> list[str]:
    with open(path) as fh:
        # Keep blank lines as empty molecules rather than silently dropping —
        # blank lines shift src/tgt alignment otherwise.
        return [line.rstrip("\n") for line in fh]


class ParallelSelfiesDataset(Dataset):
    """Pairs of (source, target) id sequences.

    Source is encoded with BOS/EOS; target likewise. The collate fn builds the
    shifted decoder input/labels.
    """

    def __init__(
        self,
        src_path: str | Path,
        tgt_path: str | Path,
        tokenizer: SelfiesTokenizer,
        max_len: int = 256,
    ):
        src_lines = _read_lines(src_path)
        tgt_lines = _read_lines(tgt_path)
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"src/tgt line count mismatch: {len(src_lines)} vs {len(tgt_lines)}")
        # Tokenize once up front (immutable) rather than re-encoding every epoch.
        self.pairs: list[tuple[list[int], list[int]]] = [
            (tokenizer.encode(s)[:max_len], tokenizer.encode(t)[:max_len])
            for s, t in zip(src_lines, tgt_lines, strict=True)
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.pairs[idx]


class Collator:
    """Pads a batch and produces (src, tgt_in, tgt_out) tensors + pad masks."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
        src = pad_sequences([b[0] for b in batch], self.pad_id)
        tgt = pad_sequences([b[1] for b in batch], self.pad_id)
        # Teacher forcing: decoder sees tgt[:-1], predicts tgt[1:].
        tgt_in = tgt[:, :-1].contiguous()
        tgt_out = tgt[:, 1:].contiguous()
        return {
            "src": src,
            "src_pad_mask": src.eq(self.pad_id),
            "tgt_in": tgt_in,
            "tgt_in_pad_mask": tgt_in.eq(self.pad_id),
            "tgt_out": tgt_out,
        }


class Seq2SeqDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer: SelfiesTokenizer | None = None
        self.train_ds: ParallelSelfiesDataset | None = None
        self.valid_ds: ParallelSelfiesDataset | None = None

    def _path(self, name: str) -> Path:
        return Path(self.cfg.data_dir) / name

    def setup(self, stage: str | None = None) -> None:
        self.tokenizer = SelfiesTokenizer.load(self._path(self.cfg.vocab_file))
        collate_max = self.cfg.max_len
        self.train_ds = ParallelSelfiesDataset(
            self._path(self.cfg.train_src),
            self._path(self.cfg.train_tgt),
            self.tokenizer,
            max_len=collate_max,
        )
        self.valid_ds = ParallelSelfiesDataset(
            self._path(self.cfg.valid_src),
            self._path(self.cfg.valid_tgt),
            self.tokenizer,
            max_len=collate_max,
        )

    @property
    def collate_fn(self) -> Collator:
        if self.tokenizer is None:
            raise RuntimeError("DataModule.setup() must be called before collate_fn")
        return Collator(self.tokenizer.pad_id)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "call setup() first"
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.valid_ds is not None, "call setup() first"
        return DataLoader(
            self.valid_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.cfg.num_workers > 0,
        )
