import pytest

torch = pytest.importorskip("torch")

from bbrt.config import ModelConfig, OptimConfig  # noqa: E402
from bbrt.models.transformer import TransformerSeq2Seq  # noqa: E402


def _tiny_cfg():
    return ModelConfig(
        d_model=32,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_seq_len=64,
    )


def test_forward_shape():
    m = TransformerSeq2Seq(_tiny_cfg(), vocab_size=20, pad_id=0).eval()
    src = torch.randint(1, 20, (3, 7))
    tgt = torch.randint(1, 20, (3, 5))
    out = m(src, tgt, src.eq(0), tgt.eq(0))
    assert out.shape == (3, 5, 20)
    assert torch.isfinite(out).all()


def test_tied_embeddings():
    m = TransformerSeq2Seq(_tiny_cfg(), vocab_size=20, pad_id=0)
    assert m.lm_head.weight is m.embed.weight


def test_padding_does_not_change_unpadded_rows():
    """Right-padding a shorter source must not alter the other rows' logits."""
    m = TransformerSeq2Seq(_tiny_cfg(), vocab_size=20, pad_id=0).eval()
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    tgt = torch.tensor([[1, 2, 3], [1, 4, 5]])
    with torch.no_grad():
        base = m(src, tgt, src.eq(0), tgt.eq(0))
    # Pad row 0's source with an extra pad column.
    src_pad = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 8, 9]])
    with torch.no_grad():
        padded = m(src_pad, tgt, src_pad.eq(0), tgt.eq(0))
    assert torch.allclose(base[0], padded[0], atol=1e-5)


def test_lightning_step_runs():
    L = pytest.importorskip("lightning")  # noqa: F841
    from bbrt.models.lit_module import LitSeq2Seq

    lit = LitSeq2Seq(
        _tiny_cfg(), OptimConfig(max_steps=10, warmup_steps=2), vocab_size=20, pad_id=0
    )
    batch = {
        "src": torch.randint(1, 20, (4, 6)),
        "src_pad_mask": torch.zeros(4, 6, dtype=torch.bool),
        "tgt_in": torch.randint(1, 20, (4, 5)),
        "tgt_in_pad_mask": torch.zeros(4, 5, dtype=torch.bool),
        "tgt_out": torch.randint(1, 20, (4, 5)),
    }
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    opt = lit.configure_optimizers()["optimizer"]
    assert opt is not None
