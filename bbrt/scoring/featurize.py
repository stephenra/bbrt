"""Shared molecular featurization — the single source of Morgan fingerprints.

Previously the Morgan-fingerprint call was duplicated across ``properties.py``,
``diversity.py`` and the DRD2 scorer. This module centralizes it and uses RDKit's
modern ``MorganGenerator`` API (the old ``GetMorganFingerprintAsBitVect`` is
deprecated). Generators are cached per ``(radius, n_bits)``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def mol_from_smiles(smiles: str | None) -> Any | None:
    """Parse SMILES to an RDKit ``Mol``; ``None`` on empty/invalid input."""
    if not smiles:
        return None
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=8)
def _generator(radius: int, n_bits: int):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_bitvect(mol: Any, radius: int = 2, n_bits: int = 2048):
    """Morgan/ECFP fingerprint as an RDKit ``ExplicitBitVect``."""
    return _generator(radius, n_bits).GetFingerprint(mol)


def morgan_bitvect_from_smiles(smiles: str | None, radius: int = 2, n_bits: int = 2048):
    """Convenience: SMILES -> ``ExplicitBitVect`` (``None`` if unparseable)."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return morgan_bitvect(mol, radius, n_bits)
