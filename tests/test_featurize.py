import pytest

pytest.importorskip("rdkit")

from rdkit import DataStructs  # noqa: E402

from bbrt.scoring.featurize import (  # noqa: E402
    mol_from_smiles,
    morgan_bitvect,
    morgan_bitvect_from_smiles,
)


def test_mol_from_smiles():
    assert mol_from_smiles("CCO") is not None
    assert mol_from_smiles("not_a_smiles") is None
    assert mol_from_smiles(None) is None
    assert mol_from_smiles("") is None


def test_self_similarity_is_one():
    fp = morgan_bitvect_from_smiles("c1ccccc1")
    assert fp is not None
    assert DataStructs.TanimotoSimilarity(fp, fp) == pytest.approx(1.0)


def test_invalid_returns_none():
    assert morgan_bitvect_from_smiles("xxx") is None


def test_radius_nbits_configurable():
    mol = mol_from_smiles("CCO")
    fp = morgan_bitvect(mol, radius=3, n_bits=1024)
    assert fp.GetNumBits() == 1024
