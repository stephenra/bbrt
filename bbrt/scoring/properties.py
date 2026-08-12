"""Molecular property functions and a scorer registry for BBRT.

Ported from ``props/properties.py`` + ``data_analysis/mmpa.py``. All scoring
functions take a SMILES string and return a float (or ``None`` where the
original returned ``None`` for invalid input). ``get_scorer`` maps a config name
(``logp04`` / ``qed`` / ``drd2``) to a callable.
"""

from __future__ import annotations

from collections.abc import Callable

from bbrt.scoring import drd2_scorer, sascorer
from bbrt.scoring.featurize import mol_from_smiles as _mol
from bbrt.scoring.featurize import morgan_bitvect

# Normalization constants for penalized logP (from the graph-to-graph reference,
# Jin et al. 2018). Kept verbatim to preserve numeric parity with prior work.
_LOGP_MEAN = 2.4570953396190123
_LOGP_STD = 1.434324401111988
_SA_MEAN = -3.0525811293166134
_SA_STD = 0.8335207024513095
_CYCLE_MEAN = -0.0485696876403053
_CYCLE_STD = 0.2860212110245455


def selfies_to_smiles(selfies_str: str | None) -> str | None:
    """Decode a SELFIES string (raw or space-tokenized) to SMILES.

    Returns ``None`` if decoding fails or yields an empty molecule.
    """
    if not selfies_str:
        return None
    import selfies as sf

    try:
        smiles = sf.decoder(selfies_str.replace(" ", ""))
    except Exception:
        return None
    return smiles or None


def similarity(a: str | None, b: str | None) -> float:
    """Tanimoto similarity of Morgan (r=2, 2048-bit) fingerprints (0.0 if invalid)."""
    from rdkit import DataStructs

    amol, bmol = _mol(a), _mol(b)
    if amol is None or bmol is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(morgan_bitvect(amol), morgan_bitvect(bmol))


def qed(smiles: str | None) -> float:
    from rdkit.Chem import QED

    mol = _mol(smiles)
    if mol is None:
        return 0.0
    return QED.qed(mol)


def drd2(smiles: str | None) -> float:
    """DRD2 activity probability in [0, 1] (Transformer-encoder classifier)."""
    if not smiles:
        return 0.0
    return drd2_scorer.get_score(smiles)


def penalized_logp(smiles: str | None) -> float | None:
    """logP penalized by synthetic accessibility and macrocycle penalty."""
    import networkx as nx
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = _mol(smiles)
    if mol is None:
        return None

    log_p = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
    sa = -sascorer.calculate_score(mol)

    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    cycle_length = 0 if not cycle_list else max(len(j) for j in cycle_list)
    cycle_length = 0 if cycle_length <= 6 else cycle_length - 6
    cycle_score = -cycle_length

    norm_log_p = (log_p - _LOGP_MEAN) / _LOGP_STD
    norm_sa = (sa - _SA_MEAN) / _SA_STD
    norm_cycle = (cycle_score - _CYCLE_MEAN) / _CYCLE_STD
    return norm_log_p + norm_sa + norm_cycle


# --------------------------------------------------------------------------- #
# Scorer registry
# --------------------------------------------------------------------------- #
_SCORERS: dict[str, Callable[[str], float | None]] = {
    "logp04": penalized_logp,
    "penalized_logp": penalized_logp,
    "qed": qed,
    "drd2": drd2,
}


def get_scorer(name: str) -> Callable[[str], float | None]:
    try:
        return _SCORERS[name]
    except KeyError:
        raise ValueError(f"unknown score_func {name!r}; choices: {sorted(_SCORERS)}") from None
