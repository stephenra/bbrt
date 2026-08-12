"""The Black Box Recursive Translation (BBRT) inference loop.

Clean port of the original ``src/bbrt.py`` (which had hardcoded ``/data`` paths,
a missing ``import re``, and ``IPython.embed`` debug calls). The algorithm:

  1. Start from a set of seed molecules (as SELFIES).
  2. Translate each seed into ``n_best`` candidates (beam or stochastic top-k).
  3. Score every candidate by the target property and keep the best per seed.
  4. Feed the kept candidates back in as the next iteration's seeds.
  5. Track per-iteration population statistics and the running best.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bbrt._logging import get_logger
from bbrt.config import BBRTConfig
from bbrt.inference.decode import Generator
from bbrt.scoring.properties import get_scorer, selfies_to_smiles, similarity

logger = get_logger(__name__)


class BBRT:
    def __init__(self, generator: Generator, cfg: BBRTConfig, seeds: list[str]):
        self.gen = generator
        self.cfg = cfg
        self.seeds = seeds
        self.score_fn = get_scorer(cfg.score_func)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mean_pop: list[float] = []
        self.std_pop: list[float] = []
        self.max_pop: list[float] = []
        self.max_so_far: list[float] = []

        # Per-run memoization: SELFIES decode and property scoring are pure
        # functions of the string (the model/scorer are fixed), so each unique
        # molecule is decoded and scored at most once across the whole loop.
        self._smiles_cache: dict[str | None, str | None] = {}
        self._score_cache: dict[str, float | None] = {}
        self._sim_cache: dict[tuple[str, str], float] = {}

    # -- helpers ------------------------------------------------------------ #
    def _smiles(self, selfies_str: str | None) -> str | None:
        if selfies_str not in self._smiles_cache:
            self._smiles_cache[selfies_str] = selfies_to_smiles(selfies_str)
        return self._smiles_cache[selfies_str]

    def _similarity(self, a_smiles: str | None, b_smiles: str | None) -> float:
        if a_smiles is None or b_smiles is None:
            return 0.0
        key = (a_smiles, b_smiles)
        if key not in self._sim_cache:
            self._sim_cache[key] = similarity(a_smiles, b_smiles)
        return self._sim_cache[key]

    def _score(self, smiles: str | None) -> float | None:
        if smiles is None:
            return None
        if smiles not in self._score_cache:
            try:
                self._score_cache[smiles] = self.score_fn(smiles)
            except Exception:
                self._score_cache[smiles] = None
        return self._score_cache[smiles]

    def _record(self, smiles_list: list[str | None]) -> None:
        vals = [v for v in (self._score(s) for s in smiles_list) if v is not None]
        if not vals:
            self.mean_pop.append(float("nan"))
            self.std_pop.append(float("nan"))
            self.max_pop.append(float("-inf"))
            return
        self.mean_pop.append(float(np.mean(vals)))
        self.std_pop.append(float(np.std(vals)))
        self.max_pop.append(float(np.max(vals)))

    def _rank(self, cands: list[list[str]], prev: list[str]) -> list[str]:
        """Pick the best-scoring candidate SELFIES per seed.

        If ``cfg.similarity_min`` is set, candidates too dissimilar from the
        seed are excluded (similarity-constrained optimization). Falls back to
        the previous seed if a seed produced no eligible candidate, so the
        population size stays constant across iterations.
        """
        sim_min = self.cfg.similarity_min
        chosen: list[str] = []
        for i, cand_list in enumerate(cands):
            prev_smiles = self._smiles(prev[i]) if sim_min is not None else None
            best_selfies, best_val = None, float("-inf")
            for selfies_str in cand_list:
                cand_smiles = self._smiles(selfies_str)
                val = self._score(cand_smiles)
                if val is None:
                    continue
                if sim_min is not None and self._similarity(prev_smiles, cand_smiles) < sim_min:
                    continue
                if val > best_val:
                    best_val, best_selfies = val, selfies_str
            chosen.append(best_selfies if best_selfies is not None else prev[i])
        return chosen

    def _save(self, selfies_list: list[str], name: str) -> None:
        smiles = [self._smiles(s) for s in selfies_list]
        pd.DataFrame({"selfies": selfies_list, "smiles": smiles}).to_csv(
            self.output_dir / name, index=False
        )

    def _save_prescored(self, cands: list[list[str]], it: int) -> None:
        rows = [
            {"seed_idx": seed_idx, "selfies": c, "smiles": self._smiles(c)}
            for seed_idx, cand_list in enumerate(cands)
            for c in cand_list
        ]
        pd.DataFrame(rows).to_csv(self.output_dir / f"prescored_preds_{it}.csv", index=False)

    # -- main loop ---------------------------------------------------------- #
    def run(self) -> dict[str, list[float]]:
        src = list(self.seeds)
        self._record([self._smiles(s) for s in src])  # iter 0 = seed population

        for it in range(self.cfg.num_iters):
            logger.info("iteration %d/%d (seeds=%d)", it + 1, self.cfg.num_iters, len(src))
            cands = self.gen.translate(
                src,
                mode=self.cfg.translate_type,
                n_best=self.cfg.n_best,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
                temperature=self.cfg.temperature,
                beam_size=self.cfg.beam_size,
                max_len=self.cfg.max_decode_len,
                seed=self.cfg.seed + it,
            )
            self._save_prescored(cands, it)

            chosen = self._rank(cands, prev=src)
            self._save(chosen, f"scored_preds_{it}.csv")
            self._record([self._smiles(s) for s in chosen])
            src = chosen

        # Running best across iterations.
        running = self.max_pop[0]
        for m in self.max_pop:
            running = max(running, m)
            self.max_so_far.append(running)

        history = {
            "mean_pop": self.mean_pop,
            "std_pop": self.std_pop,
            "max_pop": self.max_pop,
            "max_so_far": self.max_so_far,
        }
        pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)
        logger.info(
            "done. final max %.4f, best-ever %.4f. outputs -> %s",
            self.max_pop[-1],
            self.max_so_far[-1],
            self.output_dir,
        )
        return history
