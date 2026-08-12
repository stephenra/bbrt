"""DataModule for the DRD2 activity classifier.

Reads a labeled CSV (SMILES + binary activity), auto-detecting the columns,
encodes SMILES -> SELFIES token ids, builds (or reuses) a vocabulary, does a
stratified train/val split, and exposes a class-imbalance ``pos_weight``.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bbrt._logging import get_logger
from bbrt.config import DRD2Config
from bbrt.data.collate import pad_sequences
from bbrt.data.process import smiles_to_selfies
from bbrt.data.tokenizer import SelfiesTokenizer

logger = get_logger(__name__)

_SMILES_NAMES = (
    "smiles",
    "canonical_smiles",
    "canonical",
    "smile",
    "mol",
    "molecule",
    "structure",
)
_LABEL_NAMES = ("activity", "label", "active", "y", "target", "class", "hit")


def detect_columns(
    df: pd.DataFrame, smiles_col: str | None, label_col: str | None
) -> tuple[str, str]:
    """Best-effort detection of the SMILES and binary-label columns.

    Falls back to the first non-numeric column for SMILES (robust across pandas
    versions, incl. 3.0's default string dtype) and the first binary 0/1 column
    for the label.
    """
    from pandas.api import types as ptypes

    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}

    if smiles_col is None:
        for cand in _SMILES_NAMES:
            if cand in lower:
                smiles_col = lower[cand]
                break
    if smiles_col is None:
        for c in cols:
            if not ptypes.is_numeric_dtype(df[c]):
                smiles_col = c
                break

    if label_col is None:
        for cand in _LABEL_NAMES:
            if cand in lower:
                label_col = lower[cand]
                break
    if label_col is None:
        for c in cols:
            if c == smiles_col:
                continue
            vals = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if 0 < len(vals) <= 2 and set(np.round(vals).astype(int)).issubset({0, 1}):
                label_col = c
                break

    if smiles_col is None or label_col is None:
        raise ValueError(
            f"could not auto-detect SMILES/label columns from {cols}; "
            "pass smiles_col/label_col explicitly"
        )
    return smiles_col, label_col


def load_labeled_csv(
    path: str | Path,
    smiles_col: str | None = None,
    label_col: str | None = None,
    max_rows: int | None = None,
) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    smiles_col, label_col = detect_columns(df, smiles_col, label_col)
    if max_rows is not None and max_rows < len(df):
        # Random subsample — the public CSV is sorted by label, so head() would
        # yield a single class.
        df = df.sample(n=max_rows, random_state=0).reset_index(drop=True)
    smiles = df[smiles_col].astype(str).tolist()
    labels = pd.to_numeric(df[label_col], errors="coerce").round().to_numpy()
    logger.info("loaded %s: %d rows (smiles=%r, label=%r)", path, len(df), smiles_col, label_col)
    return smiles, labels


def _stratified_split(y: np.ndarray, valid_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * valid_frac))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


class _SelfiesLabelDataset(Dataset):
    def __init__(
        self, selfies: list[str], labels: np.ndarray, tokenizer: SelfiesTokenizer, max_len: int
    ):
        # Tokenize once up front rather than re-encoding every epoch.
        self.samples: list[tuple[list[int], int]] = [
            (tokenizer.encode(s)[:max_len], int(lab))
            for s, lab in zip(selfies, labels, strict=True)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]


class _LabelCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[tuple[list[int], int]]) -> dict[str, torch.Tensor]:
        ids = pad_sequences([seq for seq, _ in batch], self.pad_id)
        labels = torch.tensor([lab for _, lab in batch], dtype=torch.long)
        return {"ids": ids, "pad_mask": ids.eq(self.pad_id), "label": labels}


class DRD2DataModule(L.LightningDataModule):
    def __init__(self, cfg: DRD2Config, tokenizer: SelfiesTokenizer | None = None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.pos_weight: float | None = None
        self.train_ds: _SelfiesLabelDataset | None = None
        self.val_ds: _SelfiesLabelDataset | None = None

    def _train_path(self) -> Path:
        p = Path(self.cfg.train_csv)
        return p if p.is_absolute() or p.exists() else Path(self.cfg.data_dir) / self.cfg.train_csv

    def setup(self, stage: str | None = None) -> None:
        smiles, labels = load_labeled_csv(
            self._train_path(), self.cfg.smiles_col, self.cfg.label_col, self.cfg.max_rows
        )
        selfies: list[str] = []
        y: list[int] = []
        for smi, lab in tqdm(
            zip(smiles, labels, strict=True), total=len(smiles), desc="encode SELFIES", leave=False
        ):
            if np.isnan(lab):
                continue
            enc = smiles_to_selfies(smi)
            if enc:
                selfies.append(enc)
                y.append(int(lab))
        if not selfies:
            raise ValueError("no molecules encoded to SELFIES; check the input CSV")
        y_arr = np.asarray(y)
        logger.info(
            "encoded %d molecules (%d active / %d inactive)",
            len(selfies),
            int(y_arr.sum()),
            int((y_arr == 0).sum()),
        )

        if self.tokenizer is None:
            self.tokenizer = SelfiesTokenizer.build([selfies])

        tr, va = _stratified_split(y_arr, self.cfg.valid_frac, self.cfg.seed)
        self.train_ds = _SelfiesLabelDataset(
            [selfies[i] for i in tr], y_arr[tr], self.tokenizer, self.cfg.max_len
        )
        self.val_ds = _SelfiesLabelDataset(
            [selfies[i] for i in va], y_arr[va], self.tokenizer, self.cfg.max_len
        )
        n_pos = int(y_arr[tr].sum())
        n_neg = int(len(tr) - n_pos)
        self.pos_weight = float(n_neg / max(n_pos, 1))
        logger.info("train pos_weight=%.2f (neg/pos = %d/%d)", self.pos_weight, n_neg, n_pos)

    @property
    def _collate(self) -> _LabelCollator:
        if self.tokenizer is None:
            raise RuntimeError("DRD2DataModule.setup() must be called first")
        return _LabelCollator(self.tokenizer.pad_id)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "call setup() first"
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate,
            drop_last=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "call setup() first"
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate,
            persistent_workers=self.cfg.num_workers > 0,
        )
