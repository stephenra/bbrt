"""BBRT inference entry point (replaces the hardcoded ``src/bbrt.py`` script)."""

from __future__ import annotations

from pathlib import Path

import torch

from bbrt._logging import get_logger
from bbrt.config import BBRTConfig
from bbrt.data.process import smiles_to_selfies
from bbrt.data.tokenizer import SelfiesTokenizer
from bbrt.inference.bbrt import BBRT
from bbrt.inference.decode import Generator
from bbrt.models.lit_module import LitSeq2Seq
from bbrt.scoring.diversity import diverse_subset
from bbrt.scoring.properties import selfies_to_smiles

logger = get_logger(__name__)


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        return "cpu"
    return requested


def _load_seeds(cfg: BBRTConfig) -> list[str]:
    """Return seeds as (space-tokenized) SELFIES strings."""
    lines = [ln.strip() for ln in Path(cfg.seed_file).read_text().splitlines() if ln.strip()]
    if cfg.seed_format == "smiles":
        return [enc for smi in lines if (enc := smiles_to_selfies(smi))]
    return lines


def run_bbrt(cfg: BBRTConfig) -> dict[str, list[float]]:
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    tokenizer = SelfiesTokenizer.load(cfg.vocab)
    lit = LitSeq2Seq.from_checkpoint(cfg.checkpoint, map_location=device)
    generator = Generator(lit.model, tokenizer, device=device)

    seeds = _load_seeds(cfg)
    if not seeds:
        raise ValueError(f"no usable seeds read from {cfg.seed_file}")

    if cfg.diverse_subset and cfg.num_seeds < len(seeds):
        smiles = [selfies_to_smiles(s) for s in seeds]
        idx = diverse_subset(smiles, cfg.num_seeds, seed=cfg.seed)
        seeds = [seeds[i] for i in idx]
    else:
        seeds = seeds[: cfg.num_seeds]

    logger.info(
        "%d seeds on %s; scoring by %s, mode=%s",
        len(seeds),
        device,
        cfg.score_func,
        cfg.translate_type,
    )
    bbrt = BBRT(generator, cfg, seeds)
    return bbrt.run()
