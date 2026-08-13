import pytest

pytest.importorskip("torch")
pytest.importorskip("selfies")
pytest.importorskip("rdkit")
pytest.importorskip("lightning")
import pandas as pd  # noqa: E402

from bbrt.config import DRD2Config, ModelConfig  # noqa: E402
from bbrt.data.drd2_data import DRD2DataModule, detect_columns, load_labeled_csv  # noqa: E402

_SMILES = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccncc1", "CCOC", "CCCCO"]


def _write_csv(tmp_path, name="drd2.train.csv", smiles_col="smiles", label_col="activity"):
    df = pd.DataFrame({smiles_col: _SMILES * 4, label_col: [1, 0] * 16})
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_detect_columns_by_name():
    df = pd.DataFrame({"canonical_smiles": ["CCO"], "activity": [1]})
    assert detect_columns(df, None, None) == ("canonical_smiles", "activity")


def test_detect_columns_fallback_binary():
    df = pd.DataFrame({"structure": ["CCO", "CCN"], "hitcol": [0, 1]})
    scol, lcol = detect_columns(df, None, None)
    assert scol == "structure"
    assert lcol == "hitcol"


def test_load_labeled_csv(tmp_path):
    p = _write_csv(tmp_path)
    smiles, labels = load_labeled_csv(p)
    assert len(smiles) == len(labels) == 32
    assert set(labels.tolist()) == {0.0, 1.0}


def test_datamodule_setup(tmp_path):
    p = _write_csv(tmp_path)
    cfg = DRD2Config(
        train_csv=str(p),
        batch_size=8,
        valid_frac=0.25,
        num_workers=0,
        model=ModelConfig(
            d_model=32,
            n_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=64,
            max_seq_len=64,
        ),
    )
    dm = DRD2DataModule(cfg)
    dm.setup()
    assert dm.tokenizer is not None
    assert dm.pos_weight is not None and dm.pos_weight > 0
    batch = next(iter(dm.train_dataloader()))
    assert set(batch) == {"ids", "pad_mask", "label"}
    assert batch["ids"].shape[0] == 8
    assert batch["label"].shape == (8,)
