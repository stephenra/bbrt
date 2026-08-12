"""End-to-end DRD2 smoke: train a few steps on a synthetic labeled CSV, then
score a molecule through the packaged scorer path."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("selfies")
pytest.importorskip("rdkit")
pytest.importorskip("lightning")
pytest.importorskip("torchmetrics")
import pandas as pd  # noqa: E402

from bbrt.config import DRD2Config, ModelConfig  # noqa: E402
from bbrt.train_drd2 import train_drd2  # noqa: E402

_ACTIVE = ["c1ccncc1", "c1ccc(N)cc1", "c1ccc(O)cc1", "c1ccc2ccccc2c1", "c1ccc(Cl)cc1"]
_INACTIVE = ["CCO", "CCCC", "CC(C)C", "CCOCC", "CCCCCC"]


def _write_csv(tmp_path):
    rows = [(s, 1) for s in _ACTIVE] + [(s, 0) for s in _INACTIVE]
    df = pd.DataFrame(rows * 6, columns=["smiles", "activity"])
    p = tmp_path / "drd2.train.csv"
    df.to_csv(p, index=False)
    return p


def test_drd2_train_and_score(tmp_path):
    csv = _write_csv(tmp_path)
    out = tmp_path / "drd2_out"
    cfg = DRD2Config(
        train_csv=str(csv),
        output_dir=str(out),
        batch_size=8,
        valid_frac=0.25,
        num_workers=0,
        max_steps=6,
        warmup_steps=2,
        val_check_interval=2,
        log_every_n_steps=1,
        precision="32-true",
        accelerator="cpu",
        devices=1,
        head_hidden=16,
        model=ModelConfig(
            d_model=32,
            n_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=64,
            dropout=0.0,
            max_seq_len=64,
        ),
    )
    ckpt = train_drd2(cfg)
    assert (out / "best.ckpt").exists()
    assert (out / "vocab.json").exists()
    assert (out / "metrics.json").exists()

    from bbrt.scoring import drd2_scorer

    drd2_scorer.set_checkpoint(ckpt)
    score = drd2_scorer.get_score("c1ccncc1")
    assert 0.0 <= score <= 1.0
    # invalid SMILES -> 0.0
    assert drd2_scorer.get_score("not_a_smiles") == 0.0
