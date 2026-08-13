"""LightningModule wrapping the Transformer seq2seq."""

from __future__ import annotations

import math

import lightning as L
import torch
from torch import nn

from bbrt.config import ModelConfig, OptimConfig
from bbrt.models.transformer import TransformerSeq2Seq


def _lr_lambda(step: int, warmup: int, max_steps: int, scheduler: str) -> float:
    step = max(step, 1)
    if scheduler == "inverse_sqrt":
        return min(step**-0.5, step * warmup**-1.5) * (warmup**0.5)
    if scheduler == "warmup_cosine":
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0  # "none"


def build_adamw(
    named_params,
    *,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: int,
    betas: tuple[float, float] = (0.9, 0.999),
    scheduler: str = "warmup_cosine",
) -> dict:
    """AdamW (no weight decay on 1-D params: biases + norm gains) + LR schedule.

    Returns the dict Lightning's ``configure_optimizers`` expects. Shared by the
    seq2seq and DRD2 modules so the decay policy and schedule stay in one place.
    """
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for _, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, warmup_steps, max_steps, scheduler)
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
    }


class LitSeq2Seq(L.LightningModule):
    """Label-smoothed cross-entropy training of the Transformer."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        optim_cfg: OptimConfig,
        vocab_size: int,
        pad_id: int,
    ):
        super().__init__()
        # Persist plain dicts so the checkpoint is self-describing / reloadable.
        self.save_hyperparameters(
            {
                "model_cfg": vars(model_cfg),
                "optim_cfg": vars(optim_cfg),
                "vocab_size": vocab_size,
                "pad_id": pad_id,
            }
        )
        self.optim_cfg = optim_cfg
        self.pad_id = pad_id
        self.model = TransformerSeq2Seq(model_cfg, vocab_size, pad_id=pad_id)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id, label_smoothing=optim_cfg.label_smoothing
        )

    # -- construction from a checkpoint ------------------------------------ #
    @classmethod
    def from_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> LitSeq2Seq:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hp = ckpt["hyper_parameters"]
        module = cls(
            ModelConfig(**hp["model_cfg"]),
            OptimConfig(**hp["optim_cfg"]),
            hp["vocab_size"],
            hp["pad_id"],
        )
        module.load_state_dict(ckpt["state_dict"])
        return module

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_in_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(src, tgt_in, src_pad_mask, tgt_in_pad_mask)

    # -- shared step -------------------------------------------------------- #
    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(
            batch["src"], batch["tgt_in"], batch["src_pad_mask"], batch["tgt_in_pad_mask"]
        )
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))
        with torch.no_grad():
            mask = batch["tgt_out"].ne(self.pad_id)
            correct = (logits.argmax(-1).eq(batch["tgt_out"]) & mask).sum()
            acc = correct.float() / mask.sum().clamp_min(1)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log_dict(
            {"train/loss": loss, "train/acc": acc, "train/ppl": loss.exp()},
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log_dict(
            {"val/loss": loss, "val/acc": acc, "val/ppl": loss.exp()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    # -- optim -------------------------------------------------------------- #
    def configure_optimizers(self):
        cfg = self.optim_cfg
        return build_adamw(
            self.named_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            betas=(cfg.betas[0], cfg.betas[1]),
            scheduler=cfg.scheduler,
        )
