"""Dialogue-context likelihood proxy on ESConv (Section V-G).

100 academic-pressure conversations (fixed seed) with >= 6 preceding turns, a latest seeker
utterance of >= 5 words and a supporter reply of 8-100 words as the held-out target.
Frozen Qwen2.5-0.5B-Instruct scores the target with teacher forcing under three inputs:
  L0 latest utterance; L1 preceding four turns; L2 L1 plus the situation and emotion label.
NLL per reference token (target truncated at 96 tokens); conversation-level paired bootstrap
with 10,000 resamples.

usage: python -m experiments.esconv_likelihood ESConv.json OUT.json [--seed 20260920]
"""
from __future__ import annotations
import argparse, json
import numpy as np
from .common import set_seed, save_json

PROBLEM = "academic pressure"


def select(data, n=100, seed=20260920):
    rng = np.random.default_rng(seed); items = []
    for ci, conv in enumerate(data):
        if conv.get("problem_type", "").lower() != PROBLEM:
            continue
        d = conv["dialog"]
        cand = [i for i in range(6, len(d)) if d[i]["speaker"] == "sys" and 8 <= len(d[i]["content"].split()) <= 100
                and d[i - 1]["speaker"] == "usr" and len(d[i - 1]["content"].split()) >= 5]
        if cand:
            i = int(rng.choice(cand))
            items.append(dict(conv=ci, idx=i, target=d[i]["content"], latest=d[i - 1]["content"],
                              history=d[max(0, i - 4):i], situation=conv.get("situation", ""), emotion=conv.get("emotion_type", "")))
    rng.shuffle(items)
    return items[:n]


def fmt_turns(turns): return "\n".join(("Seeker: " if t["speaker"] == "usr" else "Supporter: ") + t["content"] for t in turns)


def inputs(it):
    return {"L0": f"Seeker: {it['latest']}\nSupporter:",
            "L1": fmt_turns(it["history"]) + "\nSupporter:",
            "L2": f"Situation: {it['situation']}\nEmotion: {it['emotion']}\n" + fmt_turns(it["history"]) + "\nSupporter:"}


def nll_per_token(model, tok, prompt, target, device):
    import torch
    p = tok(prompt, return_tensors="pt").input_ids
    t = tok(" " + target, return_tensors="pt", add_special_tokens=False).input_ids[:, :96]
    ids = torch.cat([p, t], 1).to(device)
    with torch.no_grad():
        logits = model(ids).logits[0, p.shape[1] - 1:-1]
    lp = torch.log_softmax(logits.float(), -1).gather(1, t[0].to(device).unsqueeze(1)).squeeze(1)
    return float(-lp.mean())


def main(path, out, seed, model_name="Qwen/Qwen2.5-0.5B-Instruct", n_boot=10000):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name); model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    items = select(json.load(open(path)), 100, seed)
    sc = {k: np.array([nll_per_token(model, tok, inputs(it)[k], it["target"], "cpu") for it in items]) for k in ("L0", "L1", "L2")}
    rng = np.random.default_rng(seed); res = {"n": len(items), "mean_nll": {k: float(v.mean()) for k, v in sc.items()}}
    for name, d in {"L1_minus_L0": sc["L1"] - sc["L0"], "L2_minus_L1": sc["L2"] - sc["L1"]}.items():
        m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
        res[name] = dict(mean=float(d.mean()), ci=[float(np.quantile(m, .025)), float(np.quantile(m, .975))],
                         lower_in=int((d < 0).sum()))
    save_json(res, out); print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("esconv"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=20260920)
    a = ap.parse_args(); main(a.esconv, a.out, a.seed)
