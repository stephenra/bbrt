"""KV-cached incremental decoding must exactly match full recomputation."""

import pytest

torch = pytest.importorskip("torch")

from bbrt.config import ModelConfig  # noqa: E402
from bbrt.data.tokenizer import SelfiesTokenizer  # noqa: E402
from bbrt.inference.decode import Generator  # noqa: E402
from bbrt.models.transformer import TransformerSeq2Seq  # noqa: E402


def _setup():
    tok = SelfiesTokenizer.build([["[C][N][=O][Branch1][Ring1][O][C][C]"]])
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
    return Generator(model, tok, device="cpu"), tok, model


def _uncached_greedy(gen: Generator, src: list[str], max_len: int) -> list[list[str]]:
    """Reference greedy that recomputes the whole prefix each step."""
    memory, mask = gen._encode(src, max_len)
    n = memory.size(0)
    ys = torch.full((n, 1), gen.tok.bos_id, dtype=torch.long)
    finished = torch.zeros(n, dtype=torch.bool)
    collected = []
    for _ in range(max_len):
        logits = gen.model.decode(ys, memory, None, mask)[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        nxt[finished] = gen.tok.pad_id
        collected.append(nxt)
        finished |= nxt.squeeze(1).eq(gen.tok.eos_id)
        ys = torch.cat([ys, nxt], dim=1)
        if bool(finished.all()):
            break
    gy = torch.cat(collected, dim=1)
    return [[gen._ids_to_selfies(gy[i])] for i in range(n)]


def test_decode_step_matches_full_decode():
    """Per-position logits from decode_step == full decode() over a prefix."""
    _, tok, model = _setup()
    model.eval()
    gen = Generator(model, tok, device="cpu")
    memory, mask = gen._encode(["[C][N]"], max_len=32)

    prefix = [tok.bos_id, tok.stoi["[C]"], tok.stoi["[N]"], tok.stoi["[=O]"]]
    with torch.no_grad():
        full = model.decode(torch.tensor([prefix]), memory, None, mask)[0]  # (L, V)
        caches = model.init_caches()
        steps = []
        for pos, t in enumerate(prefix):
            lg = model.decode_step(torch.tensor([[t]]), memory, mask, caches, pos)
            steps.append(lg[:, -1, :])
        step_logits = torch.cat(steps, dim=0)  # (L, V)
    assert torch.allclose(full, step_logits, atol=1e-4)


def test_greedy_cached_equals_uncached():
    gen, _, model = _setup()
    model.eval()
    src = ["[C][N]", "[C][=O][Branch1]", "[O][C][C]"]
    assert gen.greedy(src, max_len=32) == _uncached_greedy(gen, src, max_len=32)


def test_beam_size_one_equals_greedy():
    gen, _, model = _setup()
    model.eval()
    src = ["[C][N]", "[O][C]"]
    beam = gen.beam_search(src, beam_size=1, n_best=1, max_len=32)
    greedy = gen.greedy(src, max_len=32)
    assert [b[0] for b in beam] == [g[0] for g in greedy]


def test_sampling_reproducible_with_cache():
    gen, _, _ = _setup()
    a = gen.sample(["[C][N]"], n_best=4, top_k=5, max_len=32, seed=7)
    b = gen.sample(["[C][N]"], n_best=4, top_k=5, max_len=32, seed=7)
    assert a == b
