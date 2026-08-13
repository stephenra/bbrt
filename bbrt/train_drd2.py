"""Training entry point for the DRD2 Transformer-encoder classifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from bbrt._logging import get_logger
from bbrt.config import DRD2Config, dump_config
from bbrt.data.drd2_data import DRD2DataModule
from bbrt.data.tokenizer import SelfiesTokenizer
from bbrt.models.drd2_model import DRD2Classifier

logger = get_logger(__name__)


def train_drd2(cfg: DRD2Config) -> str:
    """Train the DRD2 classifier. Returns the best checkpoint path."""
    L.seed_everything(cfg.seed, workers=True)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out / "config.yaml")

    # If warm-starting from a seq2seq run, reuse ITS vocab so token ids align.
    tokenizer: SelfiesTokenizer | None = None
    if cfg.init_from:
        vocab_path = Path(cfg.init_from).parent / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"warm-start needs the seq2seq vocab at {vocab_path} (next to init_from ckpt)"
            )
        tokenizer = SelfiesTokenizer.load(vocab_path)
        logger.info("reusing seq2seq vocab from %s (size %d)", vocab_path, len(tokenizer))

    dm = DRD2DataModule(cfg, tokenizer=tokenizer)
    dm.setup()
    tokenizer = dm.tokenizer
    assert tokenizer is not None and dm.train_ds is not None  # populated by setup()
    tokenizer.save(out / "vocab.json")

    model = DRD2Classifier(cfg, len(tokenizer), tokenizer.pad_id, pos_weight=dm.pos_weight)
    logger.info("classifier parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    if cfg.init_from:
        model.warm_start_from_seq2seq(cfg.init_from)

    # Clamp val cadence so tiny datasets don't exceed the batches-per-epoch limit.
    steps_per_epoch = max(1, len(dm.train_ds) // cfg.batch_size)
    val_interval = min(cfg.val_check_interval, steps_per_epoch)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(out),
        filename="drd2-step{step}",
        monitor="val/auroc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_steps=cfg.max_steps,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,  # type: ignore[arg-type]
        gradient_clip_val=cfg.grad_clip,
        val_check_interval=val_interval,
        log_every_n_steps=cfg.log_every_n_steps,
        num_sanity_val_steps=0,
        default_root_dir=str(out),
        callbacks=[ckpt_cb, lr_cb],
    )
    trainer.fit(model, datamodule=dm)

    best = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if best:
        shutil.copy(best, out / "best.ckpt")
        logger.info("best checkpoint: %s -> %s", best, out / "best.ckpt")

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("val metrics: %s", metrics)
    return str(out / "best.ckpt")
