"""DRD2 activity score from the Transformer-encoder classifier.

Loads a :class:`bbrt.models.drd2_model.DRD2Classifier` checkpoint (produced by
``bbrt drd2-train``) plus its sibling ``vocab.json`` and scores SMILES as an
activity probability in [0, 1]. The checkpoint path defaults to
``output/drd2/best.ckpt`` and can be overridden via the ``BBRT_DRD2_CKPT``
environment variable or :func:`set_checkpoint`.

This replaced the original 2017-era ECFP->SVM oracle (preserved in git history).
"""

from __future__ import annotations

import os
from pathlib import Path

from bbrt._logging import get_logger

logger = get_logger(__name__)

_DEFAULT_CKPT = "output/drd2/best.ckpt"
_scorer: Drd2Scorer | None = None


class Drd2Scorer:
    """Wraps a trained DRD2 classifier for single-SMILES scoring."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        import torch

        from bbrt.data.tokenizer import SelfiesTokenizer
        from bbrt.models.drd2_model import DRD2Classifier

        checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.model = (
            DRD2Classifier.from_checkpoint(str(checkpoint), map_location=self.device)
            .to(self.device)
            .eval()
        )
        vocab_path = checkpoint.parent / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"DRD2 vocab.json not found next to checkpoint: {vocab_path}")
        self.tok = SelfiesTokenizer.load(vocab_path)
        logger.info("loaded DRD2 classifier from %s", checkpoint)

    def score(self, smiles: str | None) -> float:
        import torch

        from bbrt.data.process import smiles_to_selfies

        enc = smiles_to_selfies(smiles) if smiles else None
        if not enc:
            return 0.0
        ids = self.tok.encode(enc)[: self.model.cfg.max_len]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        pad = x.eq(self.tok.pad_id)
        with torch.no_grad():
            logit = self.model(x, pad)
        return float(torch.sigmoid(logit).item())


def set_checkpoint(path: str | Path, device: str = "cpu") -> Drd2Scorer:
    """Load a specific DRD2 checkpoint, replacing any cached scorer."""
    global _scorer
    _scorer = Drd2Scorer(path, device=device)
    return _scorer


def _get_scorer() -> Drd2Scorer:
    global _scorer
    if _scorer is None:
        ckpt = os.environ.get("BBRT_DRD2_CKPT", _DEFAULT_CKPT)
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"DRD2 classifier checkpoint not found at '{ckpt}'. Train it first with "
                "`bbrt drd2-fetch && bbrt drd2-train`, or set BBRT_DRD2_CKPT to a trained "
                "checkpoint."
            )
        _scorer = Drd2Scorer(ckpt)
    return _scorer


def get_score(smiles: str | None) -> float:
    """DRD2 activity probability in [0, 1] (0.0 for invalid SMILES)."""
    return _get_scorer().score(smiles)
