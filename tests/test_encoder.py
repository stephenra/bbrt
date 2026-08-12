import pytest

torch = pytest.importorskip("torch")

from bbrt.config import ModelConfig  # noqa: E402
from bbrt.models.encoder import TransformerEncoder  # noqa: E402


def _cfg():
    return ModelConfig(
        d_model=32,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_seq_len=64,
    )


def test_forward_and_pool_shapes():
    enc = TransformerEncoder(_cfg(), vocab_size=20, pad_id=0).eval()
    ids = torch.randint(1, 20, (3, 7))
    pad = ids.eq(0)
    h = enc(ids, pad)
    assert h.shape == (3, 7, 32)
    pooled = enc.pooled(ids, pad)
    assert pooled.shape == (3, 32)
    assert torch.isfinite(pooled).all()


def test_pool_is_padding_invariant():
    """Mean pool over real tokens must ignore appended padding."""
    enc = TransformerEncoder(_cfg(), vocab_size=20, pad_id=0).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    ids_pad = torch.tensor([[1, 2, 3, 4, 0, 0]])
    with torch.no_grad():
        a = enc.pooled(ids, ids.eq(0))
        b = enc.pooled(ids_pad, ids_pad.eq(0))
    assert torch.allclose(a, b, atol=1e-5)
