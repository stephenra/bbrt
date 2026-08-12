"""End-to-end BBRT smoke test on a random-weight model (no training required).

Proves the full pipeline is wired: encode seeds -> translate -> decode SELFIES
-> score -> rank -> propagate -> write outputs.
"""

import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from bbrt.config import BBRTConfig, ModelConfig  # noqa: E402
from bbrt.data.tokenizer import SelfiesTokenizer  # noqa: E402
from bbrt.inference.bbrt import BBRT  # noqa: E402
from bbrt.inference.decode import Generator  # noqa: E402
from bbrt.models.transformer import TransformerSeq2Seq  # noqa: E402


def _selfies(smiles: str) -> str:
    return " ".join(sf.split_selfies(sf.encoder(smiles)))


def test_bbrt_end_to_end(tmp_path):
    smiles_seeds = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "c1ccncc1"]
    selfies_seeds = [_selfies(s) for s in smiles_seeds]

    tok = SelfiesTokenizer.build([selfies_seeds])
    cfg_model = ModelConfig(
        d_model=32,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_seq_len=128,
    )
    model = TransformerSeq2Seq(cfg_model, vocab_size=len(tok), pad_id=tok.pad_id)
    gen = Generator(model, tok, device="cpu")

    cfg = BBRTConfig(
        output_dir=str(tmp_path),
        score_func="qed",  # pure-RDKit, no SA_Score contrib needed
        translate_type="sd",
        num_iters=2,
        num_seeds=len(selfies_seeds),
        n_best=3,
        top_k=5,
        max_decode_len=48,
        diverse_subset=False,
        seed=0,
    )
    bbrt = BBRT(gen, cfg, selfies_seeds)
    history = bbrt.run()

    # iter 0 (seeds) + num_iters recorded populations
    assert len(history["max_pop"]) == cfg.num_iters + 1
    assert len(history["max_so_far"]) == cfg.num_iters + 1
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "scored_preds_0.csv").exists()
    assert (tmp_path / "prescored_preds_0.csv").exists()


def test_bbrt_beam_mode(tmp_path):
    selfies_seeds = [_selfies(s) for s in ["CCO", "CCN", "c1ccccc1"]]
    tok = SelfiesTokenizer.build([selfies_seeds])
    model = TransformerSeq2Seq(
        ModelConfig(
            d_model=32,
            n_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=64,
            dropout=0.0,
            max_seq_len=64,
        ),
        vocab_size=len(tok),
        pad_id=tok.pad_id,
    )
    gen = Generator(model, tok, device="cpu")
    cfg = BBRTConfig(
        output_dir=str(tmp_path),
        score_func="qed",
        translate_type="beam",
        num_iters=1,
        num_seeds=3,
        n_best=2,
        beam_size=4,
        max_decode_len=32,
        diverse_subset=False,
    )
    history = BBRT(gen, cfg, selfies_seeds).run()
    assert len(history["max_pop"]) == 2


def test_rank_similarity_constraint(tmp_path):
    """similarity_min excludes candidates too dissimilar from the seed."""
    from rdkit import Chem

    from bbrt.scoring.properties import qed, selfies_to_smiles, similarity

    seed = "c1ccccc1"  # benzene
    near = "Cc1ccccc1"  # toluene: similar to benzene
    far = "CC(=O)Nc1ccc(O)cc1"  # acetaminophen: dissimilar, but higher QED
    # Preconditions that make the test meaningful.
    assert similarity(seed, near) > similarity(seed, far)
    assert qed(far) > qed(near)
    threshold = (similarity(seed, near) + similarity(seed, far)) / 2

    prev = [_selfies(seed)]
    cands = [[_selfies(near), _selfies(far)]]

    def canon(s: str) -> str:
        return Chem.MolToSmiles(Chem.MolFromSmiles(s))

    unconstrained = BBRT(None, BBRTConfig(output_dir=str(tmp_path / "u"), score_func="qed"), [])
    picked_u = selfies_to_smiles(unconstrained._rank(cands, prev)[0])
    assert canon(picked_u) == canon(far)  # unconstrained picks the higher-QED (dissimilar) one

    constrained = BBRT(
        None,
        BBRTConfig(output_dir=str(tmp_path / "c"), score_func="qed", similarity_min=threshold),
        [],
    )
    picked_c = selfies_to_smiles(constrained._rank(cands, prev)[0])
    assert canon(picked_c) == canon(near)  # constrained rejects the dissimilar one
