"""SELFIES tokenizer + vocabulary.

SELFIES strings are sequences of bracketed tokens, e.g. ``[C][N][=O]``. The
original repo stored these space-separated on disk. This tokenizer builds a
vocabulary over those tokens, encodes to integer id sequences (with BOS/EOS),
and decodes back. It intentionally treats a SELFIES *token* (a full ``[...]``
symbol) as the atomic unit -- this is what makes the SELFIES grammar robust,
so we never split inside a bracket.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

# Fallback splitter if the `selfies` package is unavailable at import time.
_TOKEN_RE = re.compile(r"\[[^\]]*\]|\.")


def split_selfies(s: str) -> list[str]:
    """Split a SELFIES string into its bracketed tokens.

    Prefers ``selfies.split_selfies`` (grammar-aware); falls back to a regex.
    Accepts either a raw SELFIES string (``[C][N]``) or an already
    space-separated one (``[C] [N]``).
    """
    s = s.strip()
    if not s:
        return []
    if " " in s:  # already tokenized on disk
        return s.split()
    try:
        import selfies as sf

        return list(sf.split_selfies(s))
    except Exception:
        return _TOKEN_RE.findall(s)


class SelfiesTokenizer:
    """Maps SELFIES tokens <-> integer ids.

    Special tokens occupy fixed low ids so checkpoints and vocab files stay
    stable across rebuilds.
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    SPECIALS = (PAD, BOS, EOS, UNK)

    def __init__(self, stoi: dict[str, int]):
        self.stoi = dict(stoi)
        self.itos = {i: t for t, i in self.stoi.items()}
        for tok in self.SPECIALS:
            if tok not in self.stoi:
                raise ValueError(f"vocab missing special token {tok!r}")
        self._special_ids = frozenset(self.stoi[t] for t in self.SPECIALS)

    # -- ids for special tokens -------------------------------------------- #
    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]

    def __len__(self) -> int:
        return len(self.stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    # -- build / persist ---------------------------------------------------- #
    @classmethod
    def build(
        cls,
        corpora: Iterable[Iterable[str]],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> SelfiesTokenizer:
        """Build a vocab from one or more iterables of SELFIES strings."""
        counter: Counter[str] = Counter()
        for corpus in corpora:
            for line in corpus:
                counter.update(split_selfies(line))
        stoi = {tok: i for i, tok in enumerate(cls.SPECIALS)}
        candidates = [
            (tok, c) for tok, c in counter.most_common() if c >= min_freq and tok not in stoi
        ]
        if max_size is not None:
            candidates = candidates[: max(0, max_size - len(stoi))]
        for tok, _ in candidates:
            stoi[tok] = len(stoi)
        return cls(stoi)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"stoi": self.stoi}, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SelfiesTokenizer:
        with open(path) as fh:
            data = json.load(fh)
        return cls(data["stoi"])

    # -- encode / decode ---------------------------------------------------- #
    def encode(
        self,
        selfies_str: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in split_selfies(selfies_str)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], strip_specials: bool = True) -> str:
        """Return a *concatenated* SELFIES string (no spaces) ready for
        ``selfies.decoder``. Stops at the first EOS."""
        toks: list[str] = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if strip_specials and i in self._special_ids:
                continue
            toks.append(self.itos.get(i, self.UNK))
        return "".join(toks)

    def decode_tokens(self, ids: Sequence[int]) -> list[str]:
        """Like :meth:`decode` but returns the token list (specials stripped)."""
        out: list[str] = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in self._special_ids:
                continue
            out.append(self.itos.get(i, self.UNK))
        return out
