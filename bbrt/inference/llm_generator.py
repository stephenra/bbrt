"""LLM-backed molecular translation — a drop-in for :class:`Generator`.

Instead of decoding candidates from the trained SELFIES Transformer, this asks
an LLM (Claude by default) to propose analog molecules for each seed. It is a
*true drop-in* for the BBRT loop: it exposes the same
``translate(...) -> list[list[str]]`` contract (SELFIES in, SELFIES out), so
:class:`bbrt.inference.bbrt.BBRT` needs no changes — the LLM becomes the
proposer while the RDKit scorers remain the judge.

The provider-specific call is a small injectable ``chat_fn(system, user) -> str``
so the whole pipeline is testable without an API key. The default backend uses
the Anthropic SDK (Claude Opus 4.8) with:

* **prompt caching** on the stable system prompt — every seed in a BBRT run
  shares it (note the 4096-token minimum cacheable prefix on Opus 4.8; a short
  system prompt won't actually cache — see ``make_anthropic_chat``);
* **structured outputs** so candidates come back as clean JSON, not free text;
* **adaptive thinking** for better chemistry reasoning.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from bbrt._logging import get_logger

logger = get_logger(__name__)

# (system_prompt, user_prompt) -> completion text. The stable system prompt is
# passed on every call so the concrete backend can cache it.
ChatFn = Callable[[str, str], str]

_OBJECTIVES = {
    "logp04": (
        "increase the penalized logP (octanol-water logP adjusted for synthetic "
        "accessibility and ring complexity)"
    ),
    "qed": "increase the QED (quantitative estimate of drug-likeness)",
    "drd2": "increase predicted DRD2 (dopamine receptor D2) binding activity",
    "qed_logp": (
        "simultaneously increase BOTH the QED (drug-likeness) AND the penalized logP "
        "(lipophilicity) -- improving one at the expense of the other does not count"
    ),
}

# Structured-output schema for the default Anthropic backend: a list of SMILES.
_SCHEMA = {
    "type": "object",
    "properties": {"molecules": {"type": "array", "items": {"type": "string"}}},
    "required": ["molecules"],
    "additionalProperties": False,
}


def objective_for(score_func: str) -> str:
    """Natural-language objective for a BBRT ``score_func`` name."""
    return _OBJECTIVES.get(score_func, f"improve the {score_func} property")


# Default few-shot guidance + worked examples. Improves proposal quality by
# grounding the edit style. It also feeds prompt caching, though on Opus 4.8 the
# cache only engages past a ~4096-token prefix -- pass a larger curated
# ``few_shot`` library to actually cross that threshold.
_DEFAULT_FEWSHOT = """\
Common property-improving edits (small, bioisosteric changes that stay close to the seed):
- Add, remove, or swap a halogen (F, Cl) to tune lipophilicity and metabolic stability.
- Interconvert a carboxylic acid, ester, and primary amide (classic acid bioisosteres).
- Add or extend a small alkyl / alkoxy group (methyl, ethyl, methoxy).
- Add a substituent to an aromatic ring, or swap a phenyl for a pyridyl (aromatic-N bioisostere).

Worked examples (seed -> analogs; note the edits are small and keep the scaffold):
Seed: CCOc1ccccc1
{"molecules": ["CCOc1ccc(C)cc1", "CCOc1ccc(F)cc1", "CCCOc1ccccc1", "COc1ccccc1"]}
Seed: O=C(O)c1ccccc1
{"molecules": ["O=C(O)c1ccc(F)cc1", "O=C(OC)c1ccccc1", "O=C(N)c1ccccc1", "O=C(O)c1ccc(C)cc1"]}
Seed: Nc1ccccc1
{"molecules": ["Cc1ccc(N)cc1", "CNc1ccccc1", "CC(=O)Nc1ccccc1", "Nc1ccccc1F"]}"""


def _build_system_prompt(objective: str, similarity: float, few_shot: str) -> str:
    return (
        "You are an expert medicinal chemist performing lead optimization.\n\n"
        f"Given a seed molecule as SMILES, propose structurally similar analogs "
        f"that {objective}.\n\n"
        "Requirements:\n"
        "- Each analog must be a single valid, synthesizable, drug-like molecule "
        "written as a SMILES string.\n"
        "- Keep analogs close to the seed via small edits (add/remove/swap a "
        f"functional group, ring, or substituent); aim for Tanimoto similarity "
        f">= {similarity:g}.\n"
        "- Favor chemically diverse edits over near-duplicates of each other.\n"
        "- Do not include the seed itself, commentary, or explanations.\n\n"
        f"{few_shot}\n\n"
        'Respond with a JSON object of the form {"molecules": ["<SMILES>", ...]}.'
    )


def _parse_candidates(text: str) -> list[str]:
    """Extract candidate SMILES from a completion (JSON first, then line-based)."""
    text = text.strip()
    try:
        data = json.loads(text)
    except Exception:
        data = None
    if isinstance(data, dict) and isinstance(data.get("molecules"), list):
        return [str(x) for x in data["molecules"]]
    if isinstance(data, list):
        return [str(x) for x in data]
    # Fallback: one candidate per line, stripping bullets / numbering / backticks.
    out = []
    for line in text.splitlines():
        s = line.strip().strip("`").lstrip("-*0123456789. ").strip()
        if s:
            out.append(s)
    return out


class LLMGenerator:
    """Drop-in generator that asks an LLM for analog molecules.

    Accepts the same ``translate`` keyword surface as
    :class:`bbrt.inference.decode.Generator` (extra decode kwargs like ``mode``,
    ``top_k``, ``beam_size`` are accepted and ignored).
    """

    def __init__(
        self,
        objective: str,
        *,
        chat_fn: ChatFn | None = None,
        model: str = "claude-opus-4-8",
        similarity: float = 0.4,
        few_shot: str | None = None,
        score_fn: Callable[[str], float | None] | None = None,
        reflect_rounds: int = 0,
        max_workers: int = 8,
        max_tokens: int = 8192,
        thinking: bool = True,
    ):
        self.system = _build_system_prompt(
            objective, similarity, few_shot if few_shot is not None else _DEFAULT_FEWSHOT
        )
        # Reflective (flavor-3, OPRO-style) mode: score own proposals and re-prompt.
        self.score_fn = score_fn
        self.reflect_rounds = reflect_rounds
        self.max_workers = max_workers
        self._chat: ChatFn = (
            chat_fn
            if chat_fn is not None
            else make_anthropic_chat(
                model=model, schema=_SCHEMA, max_tokens=max_tokens, thinking=thinking
            )
        )

    @classmethod
    def for_score_func(cls, score_func: str, **kwargs) -> LLMGenerator:
        """Build from a BBRT ``score_func`` name (e.g. ``"qed"``)."""
        return cls(objective_for(score_func), **kwargs)

    def translate(
        self,
        src_selfies: list[str],
        *,
        n_best: int = 5,
        seed: int | None = None,
        **_ignored,
    ) -> list[list[str]]:
        """Propose up to ``n_best`` analog SELFIES per seed (concurrently).

        SELFIES in, SELFIES out (matching the trained-model generator). Per-seed
        calls run on a thread pool (``max_workers``); ``ThreadPoolExecutor.map``
        preserves order.
        """
        if self.max_workers > 1 and len(src_selfies) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                return list(ex.map(lambda s: self._for_seed(s, n_best), src_selfies))
        return [self._for_seed(s, n_best) for s in src_selfies]

    # -- per-seed generation ------------------------------------------------ #
    def _for_seed(self, seed_selfies: str, n_best: int) -> list[str]:
        from bbrt.scoring.properties import selfies_to_smiles

        smiles = selfies_to_smiles(seed_selfies)
        if not smiles:
            return []
        if self.reflect_rounds and self.score_fn is not None:
            proposals = self._reflect(smiles, n_best)
        else:
            proposals = self._propose(smiles, n_best)
        return self._to_selfies(proposals, n_best)

    def _propose(self, seed_smiles: str, n_best: int, feedback=None) -> list[str]:
        """One LLM call -> list of candidate SMILES (optionally with score feedback)."""
        user = f"Seed molecule: {seed_smiles}\nPropose {n_best} improved analogs."
        if feedback:
            lines = "\n".join(f"  {smi}  ->  score {sc:.3f}" for smi, sc in feedback)
            user += (
                "\n\nYou previously proposed these analogs, with their measured objective "
                f"scores (higher is better):\n{lines}\n"
                "Propose new analogs that score strictly higher than the best of these."
            )
        try:
            text = self._chat(self.system, user)
        except Exception as exc:  # one bad seed shouldn't kill the whole run
            logger.warning("LLM call failed for seed %s: %s", seed_smiles, exc)
            return []
        return _parse_candidates(text)

    def _reflect(self, seed_smiles: str, n_best: int) -> list[str]:
        """OPRO-style: propose -> score -> re-propose with feedback; keep the best."""
        pool: dict[str, float] = {}

        def add(smiles_list: list[str]) -> None:
            for smi in smiles_list:
                if smi in pool:
                    continue
                try:
                    sc = self.score_fn(smi)  # type: ignore[misc]
                except Exception:
                    sc = None
                if sc is not None:
                    pool[smi] = sc

        add(self._propose(seed_smiles, n_best))
        for _ in range(self.reflect_rounds):
            top = sorted(pool.items(), key=lambda kv: kv[1], reverse=True)[: min(5, n_best)]
            add(self._propose(seed_smiles, n_best, feedback=top))
        return [smi for smi, _ in sorted(pool.items(), key=lambda kv: kv[1], reverse=True)]

    def _to_selfies(self, smiles_list: list[str], n_best: int) -> list[str]:
        """Validate + canonicalize SMILES to SELFIES, dedup, cap at ``n_best``."""
        from bbrt.data.process import smiles_to_selfies

        out: list[str] = []
        seen: set[str] = set()
        for smi in smiles_list:
            enc = smiles_to_selfies(smi)
            if enc and enc not in seen:
                seen.add(enc)
                out.append(enc)
            if len(out) >= n_best:
                break
        return out


def make_anthropic_chat(
    *,
    model: str = "claude-opus-4-8",
    schema: dict | None = None,
    max_tokens: int = 8192,
    thinking: bool = True,
) -> ChatFn:
    """Build a ``chat_fn`` backed by the Anthropic SDK (Claude).

    Uses prompt caching on the system prompt, structured outputs (if ``schema``
    is given), and adaptive thinking. Requires ``pip install 'bbrt[llm]'`` and an
    ``ANTHROPIC_API_KEY`` in the environment.

    Caching caveat: Opus 4.8's minimum cacheable prefix is ~4096 tokens, so a
    short system prompt silently won't cache (``cache_read_input_tokens`` stays
    0). Caching pays off once the shared system prompt is large — e.g. when you
    add few-shot exemplars of good analog edits.
    """
    import anthropic

    client = anthropic.Anthropic()

    def chat(system: str, user: str) -> str:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            # Stable across every seed call in a BBRT run -> cache it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}  # Opus 4.8: adaptive only
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        # NOTE: Opus 4.8 rejects temperature/top_p/top_k — steer via the prompt.
        resp = client.messages.create(**kwargs)
        usage = resp.usage
        logger.debug(
            "anthropic usage: input=%s cache_read=%s output=%s",
            usage.input_tokens,
            getattr(usage, "cache_read_input_tokens", 0),
            usage.output_tokens,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return chat
