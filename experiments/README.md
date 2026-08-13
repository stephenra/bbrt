# Experiments: Transformer vs. Claude as the BBRT backend

Reproduces the comparison in the top-level README ("Results"): the trained
Transformer seq2seq vs. `LLMGenerator` (Claude Sonnet 4.6 and Opus 4.8 +thinking)
as the BBRT translation model, across penalized logP, QED, and a `min(QED, logP)`
multi-objective composite.

Intermediate artifacts (shared seed sets, per-leg histories) are written to
`/tmp/exp/`; the scripts are runnable as-is once the two models are trained.

## Prerequisites

```sh
uv sync --extra llm
# trained checkpoints (see the main README, "Usage"):
#   output/logp04/best.ckpt  output/logp04/vocab.json
#   output/qed/best.ckpt     output/qed/vocab.json
# graph2graph pairs under data/logp04 and data/qed
```

## Run

```sh
# 1. Free Transformer baselines (MPS/CPU): 100 seeds, 5 iters, 5 restarts.
#    Also writes the shared seed sets /tmp/exp/seeds_{logp04,qed}.json.
uv run python experiments/run_transformer.py

# 2. Paid LLM legs (needs ANTHROPIC_API_KEY). Reuses the shared seeds.
#    Sonnet: 100 seeds x2-3 restarts; Opus: 30 seeds x1; + multi-objective
#    and reflective (OPRO) legs. See the script header for the exact matrix.
#    A fail-fast auth/credit check runs first; each leg saves incrementally.
KEY="$(cat /tmp/ak)"; rm -f /tmp/ak \
  && ANTHROPIC_API_KEY="$KEY" uv run python experiments/run_llm.py

# 3. Figure -> output/ and assets/bbrt_comparison_expanded.{png,pdf}
uv run --with matplotlib python experiments/plot_comparison.py
```

## Notes

- **Never paste the API key into a prompt.** Write it from your own terminal
  (`printf %s 'sk-ant-...' > /tmp/ak && chmod 600 /tmp/ak`); the launch line
  above captures it into the process env and deletes the file.
- **Penalized logP is unbounded and reward-hackable** — long alkyl chains score
  arbitrarily high (QED ~0). The plot flags these; trust QED and the composite.
- Opus and reflective legs are single-restart (cost); Transformer is 5 restarts,
  Sonnet 2-3. Bump `N_RESTARTS` / the per-leg counts for tighter error bars.
