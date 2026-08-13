"""Synthetic accessibility (SA) score.

The original repo vendored Ertl & Schuffenhauer's ``sascorer.py`` but relied on
a ``fpscores.pkl.gz`` fragment table that was never checked in. Modern RDKit
ships the exact same scorer (and its fragment table) as a contrib module, so we
just delegate to it -- no vendored pickle required.
"""

from __future__ import annotations

import os
import sys

_sascorer = None


def _load():
    global _sascorer
    if _sascorer is not None:
        return _sascorer
    from rdkit.Chem import RDConfig

    sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer as _s  # RDKit contrib module (bundles fpscores.pkl.gz)

    _sascorer = _s
    return _sascorer


def calculate_score(mol) -> float:
    """SA score in [1, 10] (1 = easy to synthesize, 10 = hard)."""
    return _load().calculateScore(mol)


# Backwards-compatible alias with the original camelCase name.
calculateScore = calculate_score
