import pytest

pytest.importorskip("rdkit")

from bbrt.scoring.properties import (  # noqa: E402
    get_scorer,
    penalized_logp,
    qed,
    selfies_to_smiles,
    similarity,
)


def test_qed_range():
    v = qed("c1ccccc1")
    assert 0.0 <= v <= 1.0


def test_similarity_self_is_one():
    assert similarity("CCO", "CCO") == pytest.approx(1.0)


def test_similarity_invalid_is_zero():
    assert similarity("CCO", None) == 0.0
    assert similarity("not_a_smiles", "CCO") == 0.0


def test_selfies_roundtrip():
    sf = pytest.importorskip("selfies")
    encoded = sf.encoder("CCO")
    smiles = selfies_to_smiles(encoded)
    assert smiles is not None


def test_get_scorer_unknown():
    with pytest.raises(ValueError):
        get_scorer("does_not_exist")


def test_penalized_logp_runs():
    try:
        v = penalized_logp("c1ccccc1")
    except Exception as exc:  # SA_Score contrib may be unavailable in some builds
        pytest.skip(f"penalized_logp unavailable: {exc}")
    assert v is not None
