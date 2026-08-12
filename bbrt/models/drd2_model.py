"""DRD2 activity classifier: the shared SELFIES encoder + an MLP head.

Replaces the legacy ECFP->SVM pickle with an in-house model that reuses
:class:`bbrt.models.encoder.TransformerEncoder` (the same stack as the seq2seq
model), so it can optionally be **warm-started** from a trained translation
checkpoint. Trained with class-imbalance-aware BCE and reported with AUROC/AUPRC.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision

from bbrt._logging import get_logger
from bbrt.config import DRD2Config
from bbrt.models.components import AttentionPool, RMSNorm, init_transformer_weights
from bbrt.models.encoder import TransformerEncoder
from bbrt.models.lit_module import build_adamw

logger = get_logger(__name__)


class DRD2Classifier(L.LightningModule):
    def __init__(
        self,
        cfg: DRD2Config,
        vocab_size: int,
        pad_id: int,
        pos_weight: float | None = None,
    ):
        super().__init__()
        from dataclasses import asdict

        self.save_hyperparameters(
            {
                "cfg": asdict(cfg),
                "vocab_size": vocab_size,
                "pad_id": pad_id,
                "pos_weight": pos_weight,
            }
        )
        self.cfg = cfg
        self.pad_id = pad_id

        self.encoder = TransformerEncoder(cfg.model, vocab_size, pad_id=pad_id)
        d = cfg.model.d_model
        if cfg.pool not in ("attn", "mean"):
            raise ValueError(f"pool must be 'attn' or 'mean', got {cfg.pool!r}")
        self.pool = AttentionPool(d) if cfg.pool == "attn" else None
        self.head = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.apply(init_transformer_weights)

        pw = torch.tensor(float(pos_weight)) if pos_weight is not None else torch.tensor(1.0)
        self.register_buffer("pos_weight", pw)

        self.val_auroc = BinaryAUROC()
        self.val_ap = BinaryAveragePrecision()
        self.val_acc = BinaryAccuracy()

    # -- construction / transfer ------------------------------------------- #
    @classmethod
    def from_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> DRD2Classifier:
        from bbrt.config import _from_dict

        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hp = ckpt["hyper_parameters"]
        cfg = _from_dict(DRD2Config, hp["cfg"])
        module = cls(cfg, hp["vocab_size"], hp["pad_id"], pos_weight=hp.get("pos_weight"))
        module.load_state_dict(ckpt["state_dict"])
        return module

    def warm_start_from_seq2seq(self, ckpt_path: str) -> int:
        """Copy encoder weights from a trained seq2seq checkpoint.

        Requires the seq2seq run to share this classifier's vocabulary (same
        embedding size). Returns the number of tensors transferred.
        """
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        prefix = "model.enc."
        enc_sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        if not enc_sd:
            raise ValueError(f"no encoder weights (prefix {prefix!r}) found in {ckpt_path}")
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        logger.info(
            "warm-started encoder from %s: %d tensors (missing=%d, unexpected=%d)",
            ckpt_path,
            len(enc_sd),
            len(missing),
            len(unexpected),
        )
        return len(enc_sd)

    # -- forward / steps --------------------------------------------------- #
    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Return per-molecule activity logits ``(B,)``."""
        if self.pool is None:
            pooled = self.encoder.pooled(ids, pad_mask)
        else:
            pooled = self.pool(self.encoder(ids, pad_mask), pad_mask)
        return self.head(pooled).squeeze(-1)

    def _step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(batch["ids"], batch["pad_mask"])
        y = batch["label"].float()
        pos_weight = self.pos_weight
        assert isinstance(pos_weight, torch.Tensor)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        probs = torch.sigmoid(logits)
        target = y.int()
        self.val_auroc.update(probs, target)
        self.val_ap.update(probs, target)
        self.val_acc.update(probs, target)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/ap", self.val_ap.compute(), prog_bar=True)
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.val_auroc.reset()
        self.val_ap.reset()
        self.val_acc.reset()

    # -- optim ------------------------------------------------------------- #
    def configure_optimizers(self):
        cfg = self.cfg
        return build_adamw(
            self.named_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
        )
