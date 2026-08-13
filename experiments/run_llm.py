"""Paid LLM legs (Balanced tier) + the free multi-objective Transformer baseline.

Legs: comparison (logp04, qed) Sonnet x3 + Opus focused x1; multi-objective
(qed_logp) Sonnet + Opus + Transformer(qed model); reflective (logp04, focused)
Sonnet + Opus. Each leg's result is saved immediately (incremental)."""

import json
import os

from bbrt.config import BBRTConfig
from bbrt.data.tokenizer import SelfiesTokenizer
from bbrt.inference.bbrt import BBRT
from bbrt.inference.decode import Generator
from bbrt.inference.llm_generator import LLMGenerator
from bbrt.models.lit_module import LitSeq2Seq
from bbrt.scoring.properties import get_scorer

os.makedirs("/tmp/exp", exist_ok=True)


def seeds(objective, n):
    key = "logp04" if objective == "qed_logp" else objective
    return json.load(open(f"/tmp/exp/seeds_{key}.json"))[:n]


def save(tag, sds, hists):
    json.dump({"seeds": len(sds), "restarts": hists}, open(f"/tmp/exp/{tag}.json", "w"))
    print(f"saved /tmp/exp/{tag}.json", flush=True)


def run_llm(tag, model, thinking, objective, n_seeds, n_iters, n_restarts,
            sim_min=0.4, reflect_rounds=0, workers=4):
    sds = seeds(objective, n_seeds)
    hists = []
    for r in range(n_restarts):
        kw = dict(model=model, thinking=thinking, max_workers=workers)
        if reflect_rounds:
            kw["reflect_rounds"] = reflect_rounds
            kw["score_fn"] = get_scorer(objective)
        gen = LLMGenerator.for_score_func(objective, **kw)
        cfg = BBRTConfig(output_dir=f"/tmp/exp/{tag}_{r}", score_func=objective,
                         similarity_min=sim_min, num_iters=n_iters, n_best=10,
                         max_decode_len=200, diverse_subset=False, seed=r)
        h = BBRT(gen, cfg, sds).run()
        hists.append(h)
        print(f"  {tag} r{r}: mean {h['mean_pop'][0]:.2f}->{h['mean_pop'][-1]:.2f} "
              f"best {max(h['max_so_far']):.2f}", flush=True)
    save(tag, sds, hists)


# ---- FREE: Transformer(qed model) on the multi-objective, as a baseline ---- #
def run_transformer_multi():
    tok = SelfiesTokenizer.load("output/qed/vocab.json")
    gen = Generator(LitSeq2Seq.from_checkpoint("output/qed/best.ckpt", map_location="mps").model,
                    tok, device="mps")
    sds = seeds("qed_logp", 100)
    hists = []
    for r in range(3):
        cfg = BBRTConfig(output_dir=f"/tmp/exp/tmulti_{r}", score_func="qed_logp",
                         similarity_min=0.4, translate_type="sd", num_iters=5, n_best=10,
                         top_k=5, max_decode_len=200, diverse_subset=False, seed=r)
        hists.append(BBRT(gen, cfg, sds).run())
    save("transformer_qed_logp", sds, hists)
    print("  transformer_qed_logp done", flush=True)


print(">>> free Transformer multi-objective baseline", flush=True)
run_transformer_multi()

# ---- fail-fast auth check before spending on the LLM ---- #
import anthropic  # noqa: E402
anthropic.Anthropic().messages.create(
    model="claude-sonnet-4-6", max_tokens=8, messages=[{"role": "user", "content": "ping"}])
print("auth OK", flush=True)

# ---- experiment 1: logp04 + qed comparison ---- #
run_llm("sonnet_logp04", "claude-sonnet-4-6", False, "logp04", 100, 5, 3, workers=8)
run_llm("sonnet_qed",    "claude-sonnet-4-6", False, "qed",    100, 5, 3, workers=8)
run_llm("opus_logp04",   "claude-opus-4-8",   True,  "logp04", 30, 5, 1, workers=4)
run_llm("opus_qed",      "claude-opus-4-8",   True,  "qed",    30, 5, 1, workers=4)

# ---- experiment 2: multi-objective ---- #
run_llm("sonnet_qed_logp", "claude-sonnet-4-6", False, "qed_logp", 100, 5, 2, workers=8)
run_llm("opus_qed_logp",   "claude-opus-4-8",   True,  "qed_logp", 30, 5, 1, workers=4)

# ---- experiment 3: reflective (focused) ---- #
run_llm("sonnet_reflect_logp04", "claude-sonnet-4-6", False, "logp04", 30, 3, 1,
        reflect_rounds=2, workers=8)
run_llm("opus_reflect_logp04",   "claude-opus-4-8",   True,  "logp04", 20, 3, 1,
        reflect_rounds=2, workers=4)

print("ALL LLM LEGS DONE", flush=True)
