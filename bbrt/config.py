"""Typed configuration objects for BBRT.

These dataclasses replace the OpenNMT shell flags (``train.sh`` / ``preprocess.sh``)
and the block of hardcoded parameters at the bottom of the original ``src/bbrt.py``.
All configs can be loaded from / dumped to YAML.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the tokenized-SELFIES parallel corpus lives and how to batch it."""

    data_dir: str = "data"
    train_src: str = "src_train.csv"
    train_tgt: str = "tgt_train.csv"
    valid_src: str = "src_valid.csv"
    valid_tgt: str = "tgt_valid.csv"
    vocab_file: str = "vocab.json"
    max_len: int = 256
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Transformer encoder-decoder hyper-parameters.

    Defaults give a model of comparable capacity to the original 600-dim,
    2-layer BiLSTM seq2seq, but as a modern pre-norm Transformer.
    """

    d_model: int = 512
    n_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 2048  # SwiGLU inner dim (before the 2/3 scaling)
    dropout: float = 0.1
    max_seq_len: int = 512  # RoPE table size
    tie_embeddings: bool = True  # share src/tgt embeddings + tie output head
    rope_theta: float = 10000.0


@dataclass
class OptimConfig:
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 4000
    max_steps: int = 100_000
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    scheduler: str = "warmup_cosine"  # "warmup_cosine" | "inverse_sqrt" | "none"


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = "output"
    seed: int = 1
    precision: str = "bf16-mixed"  # lightning precision string; "32-true" on CPU
    accelerator: str = "auto"
    devices: str | int = "auto"
    val_check_interval: int = 1000
    log_every_n_steps: int = 50
    save_top_k: int = 3
    accumulate_grad_batches: int = 1


@dataclass
class BBRTConfig:
    """Parameters for the recursive-translation inference loop."""

    checkpoint: str = "output/best.ckpt"
    vocab: str = "output/vocab.json"  # tokenizer vocab matching the checkpoint
    seed_file: str = "seeds.csv"  # seed molecules, one per line
    seed_format: str = "selfies"  # "selfies" | "smiles"
    output_dir: str = "output/bbrt"
    score_func: str = "logp04"  # logp04 | drd2 | qed
    translate_type: str = "sd"  # "sd" (stochastic top-k) | "beam"
    num_iters: int = 3  # BBRT iterations
    num_seeds: int = 100  # seeds kept after diverse-subset selection
    n_best: int = 5  # sequences decoded per seed
    top_k: int = 5  # top-k sampler
    top_p: float = 1.0  # nucleus sampling (1.0 = disabled)
    temperature: float = 1.0  # sampling temperature
    beam_size: int = 10
    max_decode_len: int = 256
    diverse_subset: bool = True  # MaxMin diverse seed selection
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    seed: int = 1


@dataclass
class DRD2Config:
    """Config for training the DRD2 Transformer-encoder classifier."""

    # data
    data_dir: str = "data/drd2"
    train_csv: str = "drd2.train.csv"  # relative to data_dir, or absolute
    test_csv: str = "drd2.test.csv"
    smiles_col: str | None = None  # auto-detected from the header if None
    label_col: str | None = None  # auto-detected from the header if None
    valid_frac: float = 0.1
    max_len: int = 256
    batch_size: int = 128
    num_workers: int = 4
    max_rows: int | None = None  # subsample cap for quick runs

    # model (shared encoder) + classification head
    model: ModelConfig = field(default_factory=ModelConfig)
    head_hidden: int = 256
    head_dropout: float = 0.2
    # Sequence pooling for the head. "mean" is the default: in low-data / short
    # training it beats "attn" (learned-query attention pooling), which needs
    # more training to specialize but can help at full scale.
    pool: str = "mean"  # "mean" | "attn"

    # optim
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 20_000
    grad_clip: float = 1.0

    # run
    output_dir: str = "output/drd2"
    vocab_file: str = "vocab.json"  # written under output_dir
    init_from: str | None = None  # seq2seq ckpt for encoder warm-start
    seed: int = 1
    precision: str = "bf16-mixed"  # use "32-true" on CPU
    accelerator: str = "auto"
    devices: str | int = "auto"
    val_check_interval: int = 500
    log_every_n_steps: int = 50


# --------------------------------------------------------------------------- #
# YAML (de)serialization
# --------------------------------------------------------------------------- #
def _from_dict(cls: Any, data: dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a plain dict.

    ``from __future__ import annotations`` makes ``field.type`` a string, so we
    resolve real types via ``get_type_hints`` to detect nested dataclasses.
    """
    if not dataclasses.is_dataclass(cls):
        return data
    assert isinstance(cls, type)  # we only ever pass dataclass *types*, not instances
    hints = typing.get_type_hints(cls)
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = hints.get(key)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _from_dict(ftype, value)
        elif isinstance(value, list) and typing.get_origin(ftype) is tuple:
            # YAML has no tuple type; coerce list -> tuple for tuple-typed fields.
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path, cls: type):
    """Load a YAML file into a dataclass of type ``cls``."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(cls, raw)


def dump_config(cfg, path: str | Path) -> None:
    """Serialize a dataclass config to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)
