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


def _build_system_prompt(objective: str, similarity: float) -> str:
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
        max_tokens: int = 8192,
        thinking: bool = True,
    ):
        self.system = _build_system_prompt(objective, similarity)
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
        """Propose up to ``n_best`` analog SELFIES per seed.

        SELFIES in, SELFIES out (matching the trained-model generator). Seeds are
        decoded to SMILES for the prompt; proposals are validated and re-encoded
        to SELFIES with RDKit + the ``selfies`` grammar, so only valid molecules
        are returned.
        """
        from bbrt.data.process import smiles_to_selfies
        from bbrt.scoring.properties import selfies_to_smiles

        results: list[list[str]] = []
        for sf in src_selfies:
            smiles = selfies_to_smiles(sf)
            if not smiles:
                results.append([])
                continue
            user = f"Seed molecule: {smiles}\nPropose {n_best} improved analogs."
            try:
                text = self._chat(self.system, user)
            except Exception as exc:  # one bad seed shouldn't kill the whole run
                logger.warning("LLM call failed for seed %s: %s", smiles, exc)
                results.append([])
                continue

            candidates: list[str] = []
            seen: set[str] = set()
            for cand in _parse_candidates(text):
                enc = smiles_to_selfies(cand)  # validates + canonicalizes to SELFIES
                if enc and enc not in seen:
                    seen.add(enc)
                    candidates.append(enc)
                if len(candidates) >= n_best:
                    break
            results.append(candidates)
        return results


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
