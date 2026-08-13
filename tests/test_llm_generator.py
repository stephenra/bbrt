"""LLMGenerator: drop-in for the trained-model generator, tested with an
injectable fake chat function (no API key, no network)."""

import json

import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")
pytest.importorskip("torch")  # BBRT import pulls the decode module transitively

import selfies as sf  # noqa: E402

from bbrt.config import BBRTConfig  # noqa: E402
from bbrt.inference.bbrt import BBRT  # noqa: E402
from bbrt.inference.llm_generator import (  # noqa: E402
    LLMGenerator,
    _parse_candidates,
    objective_for,
)


def _selfies(smiles: str) -> str:
    return " ".join(sf.split_selfies(sf.encoder(smiles)))


def test_objective_for():
    assert "logP" in objective_for("logp04")
    assert "QED" in objective_for("qed")
    assert "custom" in objective_for("custom")  # unknown -> generic phrasing
    assert "BOTH" in objective_for("qed_logp")  # multi-objective phrasing


def test_concurrency_matches_sequential():
    fake = lambda system, user: json.dumps({"molecules": ["c1ccc(O)cc1", "CCN"]})  # noqa: E731
    seeds = [_selfies(x) for x in ["CCO", "CCN", "c1ccccc1", "CCCC"]]
    seq = LLMGenerator("q", chat_fn=fake, max_workers=1).translate(seeds, n_best=2)
    par = LLMGenerator("q", chat_fn=fake, max_workers=4).translate(seeds, n_best=2)
    assert seq == par  # threaded map preserves order + results
    assert len(par) == 4


def test_reflective_calls_and_scores():
    from bbrt.scoring.properties import qed

    calls = {"n": 0}

    def counting_chat(system, user):
        calls["n"] += 1
        return json.dumps({"molecules": ["c1ccc(O)cc1", "CCN", "c1ccccc1"]})

    gen = LLMGenerator(
        "increase QED", chat_fn=counting_chat, score_fn=qed, reflect_rounds=2, max_workers=1
    )
    out = gen.translate([_selfies("CCO")], n_best=3)
    assert calls["n"] == 3  # 1 initial proposal + 2 reflection rounds
    assert len(out) == 1 and len(out[0]) >= 1


def test_parse_candidates_json_and_lines():
    assert _parse_candidates('{"molecules": ["CCO", "CCN"]}') == ["CCO", "CCN"]
    assert _parse_candidates('["CCO", "CCN"]') == ["CCO", "CCN"]
    # free-text fallback: one per line, bullets/numbering stripped
    assert _parse_candidates("1. CCO\n- CCN\n`c1ccccc1`") == ["CCO", "CCN", "c1ccccc1"]


def test_translate_validates_dedups_and_caps():
    def fake_chat(system: str, user: str) -> str:
        # Two valid (one repeated) + one invalid SMILES.
        return json.dumps({"molecules": ["c1ccccc1O", "not_a_smiles", "c1ccccc1O", "CCN"]})

    gen = LLMGenerator("increase QED", chat_fn=fake_chat)
    out = gen.translate([_selfies("CCO")], n_best=5)

    assert len(out) == 1
    cands = out[0]
    assert len(cands) == 2  # invalid dropped, duplicate collapsed
    # returned candidates are valid SELFIES that round-trip to SMILES
    from bbrt.scoring.properties import selfies_to_smiles

    assert all(selfies_to_smiles(c) is not None for c in cands)


def test_translate_ignores_decode_kwargs_and_bad_seed():
    gen = LLMGenerator("increase QED", chat_fn=lambda s, u: '{"molecules": ["CCO"]}')
    # BBRT passes mode/top_k/top_p/temperature/beam_size/max_len — all ignored.
    out = gen.translate(
        [_selfies("CCN"), "[Xx] [Yy]"],  # second seed is undecodable
        mode="sd",
        n_best=3,
        top_k=5,
        top_p=0.9,
        temperature=0.7,
        beam_size=4,
        max_len=64,
        seed=1,
    )
    assert len(out) == 2
    assert len(out[0]) == 1
    assert out[1] == []  # bad seed -> no candidates (BBRT falls back to the seed)


def test_drop_in_through_real_bbrt_loop(tmp_path):
    """The whole point: run the actual BBRT loop with the LLM backend."""
    analogs = ["c1ccc(O)cc1", "CC(=O)Nc1ccccc1", "CCOc1ccccc1", "c1ccncc1"]
    fake = json.dumps({"molecules": analogs})
    gen = LLMGenerator.for_score_func("qed", chat_fn=lambda system, user: fake)

    seeds = [_selfies(s) for s in ["CCO", "CCN", "c1ccccc1"]]
    cfg = BBRTConfig(
        output_dir=str(tmp_path),
        score_func="qed",
        num_iters=2,
        num_seeds=len(seeds),
        n_best=3,
        diverse_subset=False,
    )
    history = BBRT(gen, cfg, seeds).run()  # BBRT itself is unchanged

    assert len(history["max_pop"]) == cfg.num_iters + 1
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "scored_preds_0.csv").exists()
