import pytest

torch = pytest.importorskip("torch")

from bbrt.config import ModelConfig  # noqa: E402
from bbrt.data.tokenizer import SelfiesTokenizer  # noqa: E402
from bbrt.inference.decode import Generator, _filter_top_k_top_p  # noqa: E402
from bbrt.models.transformer import TransformerSeq2Seq  # noqa: E402


def _setup():
    tok = SelfiesTokenizer.build([["[C][N][=O][Branch1][Ring1][O]"]])
    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_seq_len=64,
    )
    model = TransformerSeq2Seq(cfg, vocab_size=len(tok), pad_id=tok.pad_id)
    return Generator(model, tok, device="cpu"), tok


def test_top_k_filter_keeps_k():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    filtered = _filter_top_k_top_p(logits, top_k=2)
    assert torch.isinf(filtered).sum() == 3  # 5 - 2 kept


def test_sample_shapes():
    gen, tok = _setup()
    out = gen.sample(["[C][N]", "[C][=O]"], n_best=4, top_k=3, max_len=16, seed=0)
    assert len(out) == 2
    assert all(len(cands) == 4 for cands in out)
    assert all(isinstance(s, str) for cands in out for s in cands)


def test_sample_is_seeded():
    gen, _ = _setup()
    a = gen.sample(["[C][N]"], n_best=3, top_k=5, max_len=16, seed=123)
    b = gen.sample(["[C][N]"], n_best=3, top_k=5, max_len=16, seed=123)
    assert a == b


def test_beam_shapes():
    gen, _ = _setup()
    out = gen.beam_search(["[C][N]", "[C][=O]"], beam_size=5, n_best=3, max_len=16)
    assert len(out) == 2
    assert all(len(cands) == 3 for cands in out)
