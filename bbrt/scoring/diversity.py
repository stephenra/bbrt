"""MaxMin diverse-subset selection over molecular fingerprints.

Ported from ``return_diverse_subset`` in the original ``src/bbrt.py``.
"""

from __future__ import annotations

from bbrt.scoring.featurize import morgan_bitvect_from_smiles


def diverse_subset(smiles_list: list[str | None], num: int, seed: int = 0) -> list[int]:
    """Pick ``num`` maximally diverse molecules via RDKit's MaxMin picker.

    Returns the selected indices into ``smiles_list``. Molecules that fail to
    produce a fingerprint are skipped.
    """
    from rdkit import DataStructs
    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

    valid_idx, fps = [], []
    for i, smi in enumerate(smiles_list):
        fp = morgan_bitvect_from_smiles(smi)
        if fp is not None:
            valid_idx.append(i)
            fps.append(fp)

    if num >= len(fps):
        return valid_idx

    def dist(i: int, j: int) -> float:
        return 1.0 - DataStructs.DiceSimilarity(fps[i], fps[j])

    picker = MaxMinPicker()
    picks = picker.LazyPick(dist, len(fps), num, seed=seed)
    return [valid_idx[p] for p in picks]
