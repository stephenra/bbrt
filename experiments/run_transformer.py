"""Transformer baseline legs: BBRT with the trained logp04 + QED models,
100 seeds, 5 iterations, 5 restarts each. Saves seed sets (shared with the LLM
legs) and per-restart histories."""

import json
import os

from bbrt.config import BBRTConfig
from bbrt.data.tokenizer import SelfiesTokenizer
from bbrt.inference.bbrt import BBRT
from bbrt.inference.decode import Generator
from bbrt.models.lit_module import LitSeq2Seq
from bbrt.scoring.diversity import diverse_subset
from bbrt.scoring.properties import selfies_to_smiles

os.makedirs("/tmp/exp", exist_ok=True)
N_SEEDS, N_ITERS, N_BEST, N_RESTARTS = 100, 5, 10, 5
DATA = {"logp04": "data/logp04", "qed": "data/qed", "qed_logp": "data/logp04"}


def get_seeds(objective: str) -> list[str]:
    """Diverse 100 seeds from the objective's source molecules, cached to disk."""
    path = f"/tmp/exp/seeds_{objective}.json"
    if os.path.exists(path):
        return json.load(open(path))
    sf = [ln.strip() for ln in open(f"{DATA[objective]}/src_valid.csv") if ln.strip()]
    smi = [selfies_to_smiles(s) for s in sf]
    idx = diverse_subset(smi, N_SEEDS, seed=1)
    seeds = [sf[i] for i in idx]
    json.dump(seeds, open(path, "w"))
    return seeds


def run(ckpt: str, objective: str, sim_min: float | None = 0.4) -> None:
    tok = SelfiesTokenizer.load(f"{os.path.dirname(ckpt)}/vocab.json")
    gen = Generator(LitSeq2Seq.from_checkpoint(ckpt, map_location="mps").model, tok, device="mps")
    seeds = get_seeds(objective)
    hists = []
    for r in range(N_RESTARTS):
        cfg = BBRTConfig(
            output_dir=f"/tmp/exp/t_{objective}_{r}", score_func=objective,
            similarity_min=sim_min, translate_type="sd", num_iters=N_ITERS,
            n_best=N_BEST, top_k=5, max_decode_len=200, diverse_subset=False, seed=r,
        )
        h = BBRT(gen, cfg, seeds).run()
        hists.append(h)
        print(f"  {objective} r{r}: mean {h['mean_pop'][0]:.2f}->{h['mean_pop'][-1]:.2f} "
              f"best {max(h['max_so_far']):.2f}", flush=True)
    json.dump({"seeds": len(seeds), "restarts": hists},
              open(f"/tmp/exp/transformer_{objective}.json", "w"))
    print(f"saved /tmp/exp/transformer_{objective}.json", flush=True)


print(">>> Transformer logp04 baseline (5 restarts)", flush=True)
run("output/logp04/best.ckpt", "logp04", 0.4)
print(">>> Transformer QED baseline (5 restarts)", flush=True)
run("output/qed/best.ckpt", "qed", 0.4)
print("TRANSFORMER BASELINES DONE", flush=True)
