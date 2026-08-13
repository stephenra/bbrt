"""Fetch the public DRD2 activity dataset.

Downloads the DRD2 train/test CSVs from MolecularAI/ReinventCommunity (ExCAPE-DB
-> Olivecrona lineage; SMILES + binary activity label). These are the same
family of labels the original ECFP->SVM oracle was built on. Use ``--data`` on
``bbrt drd2-train`` to train on your own CSV instead.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from bbrt._logging import get_logger

logger = get_logger(__name__)

_BASE = "https://raw.githubusercontent.com/MolecularAI/ReinventCommunity/master/notebooks/data"
FILES = {
    "drd2.train.csv": f"{_BASE}/drd2.train.csv",
    "drd2.test.csv": f"{_BASE}/drd2.test.csv",
}


def _download(url: str, dest: Path) -> None:
    logger.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "bbrt-drd2-fetch"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:  # noqa: S310 (pinned https)
        shutil.copyfileobj(resp, fh)


def fetch(out_dir: str | Path = "data/drd2", force: bool = False) -> dict[str, str]:
    """Download the DRD2 CSVs into ``out_dir``; skips files that already exist."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, url in FILES.items():
        dest = out / name
        if dest.exists() and not force:
            logger.info("%s already exists; skipping (use force=True to re-download)", dest)
        else:
            _download(url, dest)
        written[name] = str(dest)

    # Validate the header so failures surface now, not mid-training.
    import pandas as pd

    from bbrt.data.drd2_data import detect_columns

    head = pd.read_csv(written["drd2.train.csv"], nrows=100)
    scol, lcol = detect_columns(head, None, None)
    logger.info(
        "validated %s: columns=%s -> smiles=%r label=%r",
        written["drd2.train.csv"],
        list(head.columns),
        scol,
        lcol,
    )
    return written
