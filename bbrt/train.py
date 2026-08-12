"""Training entry point (replaces the OpenNMT ``train.sh`` invocation)."""

from __future__ import annotations

import shutil
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from bbrt._logging import get_logger
from bbrt.config import TrainConfig, dump_config
from bbrt.data.datamodule import Seq2SeqDataModule
from bbrt.models.lit_module import LitSeq2Seq

logger = get_logger(__name__)


def train(cfg: TrainConfig) -> str:
    """Train the Transformer seq2seq. Returns the best checkpoint path."""
    L.seed_everything(cfg.seed, workers=True)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, output_dir / "config.yaml")

    dm = Seq2SeqDataModule(cfg.data)
    dm.setup()
    tokenizer = dm.tokenizer
    assert tokenizer is not None

    # Keep the vocab next to the checkpoints so inference can find it.
    shutil.copy(Path(cfg.data.data_dir) / cfg.data.vocab_file, output_dir / "vocab.json")

    lit = LitSeq2Seq(cfg.model, cfg.optim, len(tokenizer), tokenizer.pad_id)
    logger.info("model parameters: %s", f"{lit.model.num_parameters():,}")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="seq2seq-step{step}",
        monitor="val/loss",
        mode="min",
        save_top_k=cfg.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_steps=cfg.optim.max_steps,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,  # type: ignore[arg-type]
        gradient_clip_val=cfg.optim.grad_clip,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=cfg.log_every_n_steps,
        default_root_dir=str(output_dir),
        callbacks=[ckpt_cb, lr_cb],
    )
    trainer.fit(lit, datamodule=dm)

    best = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if best:
        shutil.copy(best, output_dir / "best.ckpt")
        logger.info("best checkpoint: %s -> %s", best, output_dir / "best.ckpt")
    return str(output_dir / "best.ckpt")
