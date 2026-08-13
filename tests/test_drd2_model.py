import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("torchmetrics")

from bbrt.config import DRD2Config, ModelConfig, OptimConfig  # noqa: E402
from bbrt.models.drd2_model import DRD2Classifier  # noqa: E402
from bbrt.models.lit_module import LitSeq2Seq  # noqa: E402


def _model_cfg():
    return ModelConfig(
        d_model=32,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_seq_len=64,
    )


def _cfg():
    return DRD2Config(model=_model_cfg(), head_hidden=16, max_steps=10, warmup_steps=2)


def test_forward_shape():
    m = DRD2Classifier(_cfg(), vocab_size=20, pad_id=0, pos_weight=3.0).eval()
    ids = torch.randint(1, 20, (4, 7))
    out = m(ids, ids.eq(0))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_attention_pooling_forward():
    cfg = DRD2Config(model=_model_cfg(), head_hidden=16, pool="attn")
    m = DRD2Classifier(cfg, vocab_size=20, pad_id=0).eval()
    assert m.pool is not None
    ids = torch.randint(1, 20, (4, 7))
    out = m(ids, ids.eq(0))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_training_step_backward():
    m = DRD2Classifier(_cfg(), vocab_size=20, pad_id=0, pos_weight=2.0)
    batch = {
        "ids": torch.randint(1, 20, (4, 6)),
        "pad_mask": torch.zeros(4, 6, dtype=torch.bool),
        "label": torch.tensor([0, 1, 1, 0]),
    }
    loss = m.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_warm_start_from_seq2seq(tmp_path):
    lit = LitSeq2Seq(
        _model_cfg(), OptimConfig(max_steps=10, warmup_steps=2), vocab_size=20, pad_id=0
    )
    ckpt = tmp_path / "seq2seq.ckpt"
    torch.save({"state_dict": lit.state_dict()}, ckpt)

    m = DRD2Classifier(_cfg(), vocab_size=20, pad_id=0)
    n = m.warm_start_from_seq2seq(str(ckpt))
    assert n > 0
    # Encoder weights should now match the seq2seq encoder.
    assert torch.allclose(m.encoder.embed.weight, lit.model.enc.embed.weight)
    assert torch.allclose(m.encoder.layers[0].norm1.weight, lit.model.enc.layers[0].norm1.weight)


def test_from_checkpoint_roundtrip(tmp_path):
    m = DRD2Classifier(_cfg(), vocab_size=20, pad_id=0, pos_weight=2.5)
    ckpt = tmp_path / "drd2.ckpt"
    torch.save({"state_dict": m.state_dict(), "hyper_parameters": dict(m.hparams)}, ckpt)

    m2 = DRD2Classifier.from_checkpoint(str(ckpt))
    ids = torch.randint(1, 20, (2, 5))
    pad = ids.eq(0)
    m.eval()
    m2.eval()
    with torch.no_grad():
        assert torch.allclose(m(ids, pad), m2(ids, pad), atol=1e-5)
