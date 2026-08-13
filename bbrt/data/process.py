"""Convert SMILES training pairs to a tokenized-SELFIES parallel corpus.

Port of the original ``process_data.py`` to the modern ``selfies`` API (>=2.0,
where ``encoder`` raises instead of returning ``-1``). Given a whitespace- or
comma-separated ``pairs`` file with two SMILES columns (source, target), this:

  * encodes both columns to SELFIES,
  * drops pairs where either molecule fails to encode,
  * writes space-tokenized ``src_/tgt_`` train/valid CSVs, and
  * builds and saves the shared vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bbrt._logging import get_logger
from bbrt.data.tokenizer import SelfiesTokenizer

logger = get_logger(__name__)


def smiles_to_selfies(smiles: str) -> str | None:
    """Encode one SMILES to a space-separated SELFIES token string.

    Returns ``None`` if the molecule cannot be encoded.
    """
    import selfies as sf

    try:
        enc = sf.encoder(smiles)
    except Exception:
        return None
    if enc is None or enc == -1:
        return None
    return " ".join(sf.split_selfies(enc))


def read_pairs(path: str | Path, sep: str | None = None, shuffle: bool = True) -> pd.DataFrame:
    """Read a two-column (src, tgt) SMILES pairs file and encode to SELFIES.

    ``sep=None`` lets pandas sniff whitespace/comma. Rows with an unencodable
    source or target are dropped.
    """
    df = pd.read_csv(path, sep=sep, header=None, engine="python").iloc[:, :2]
    df.columns = ["src", "tgt"]
    if shuffle:
        df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    df["src"] = df["src"].map(smiles_to_selfies)
    df["tgt"] = df["tgt"].map(smiles_to_selfies)
    return df.dropna().reset_index(drop=True)


def train_valid_split(
    df: pd.DataFrame, valid_frac: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * (1.0 - valid_frac))
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def process(
    pairs_file: str | Path,
    output_dir: str | Path,
    valid_frac: float = 0.1,
    min_freq: int = 1,
) -> dict[str, str]:
    """Full processing pipeline. Returns a dict of written file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_pairs(pairs_file)
    if df.empty:
        raise ValueError(f"No valid SELFIES pairs produced from {pairs_file}")
    train, valid = train_valid_split(df, valid_frac)

    def _write(series: pd.Series, name: str) -> str:
        p = output_dir / name
        series.to_csv(p, index=False, header=False)
        return str(p)

    written = {
        "train_src": _write(train["src"], "src_train.csv"),
        "train_tgt": _write(train["tgt"], "tgt_train.csv"),
        "valid_src": _write(valid["src"], "src_valid.csv"),
        "valid_tgt": _write(valid["tgt"], "tgt_valid.csv"),
    }

    # Shared vocabulary over all splits & sides (matches original -share_vocab).
    tok = SelfiesTokenizer.build(
        [train["src"], train["tgt"], valid["src"], valid["tgt"]], min_freq=min_freq
    )
    vocab_path = output_dir / "vocab.json"
    tok.save(vocab_path)
    written["vocab"] = str(vocab_path)

    logger.info(
        "processed %d pairs -> %d train / %d valid; vocab size %d. written to %s",
        len(df),
        len(train),
        len(valid),
        len(tok),
        output_dir,
    )
    return written
