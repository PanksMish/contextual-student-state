"""Generate the pilot and sensor-study replies from the frozen tables (Section VI).

Reads scenarios.csv (pilot) and sensor_sessions.csv (sensor study), builds each condition's prompt with
css.conditions, generates with a fixed model (temperature 0.7, top-p 0.9, <=150 words, seed 20261008),
and writes responses.csv with the prompt SHA-256 and a blind code per response.

Run only after the pre-registration deposit; check_dataset.py verifies freeze-before-generation.

usage: python -m experiments.generate_replies DATA_DIR [--model Qwen/Qwen2.5-7B-Instruct] [--revision HASH]
"""
from __future__ import annotations
import argparse, hashlib, random, string
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from css.conditions import prompt, sensor_prompt, SYSTEM
from .common import set_seed

SEED = 20261008
ABL = {"no_history": "history_summary", "no_academic": "academic_context", "no_interv": "prior_intervention"}


def ablation_prompt(kind, sc):
    sc = dict(sc); sc[ABL[kind]] = "not provided"
    if kind == "no_interv": sc["prior_reaction"] = ""
    return prompt("C2", sc)


def main(data_dir, model_name, revision):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = Path(data_dir); sc = pd.read_csv(d / "scenarios.csv", dtype=str).fillna("")
    se = pd.read_csv(d / "sensor_sessions.csv", dtype=str).fillna("") if (d / "sensor_sessions.csv").exists() else pd.DataFrame()
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype="auto").eval()
    jobs = []
    for _, r in sc.iterrows():
        for c in ("C0", "C1", "CF", "C2"): jobs.append(("pilot", r.scenario_id, c, prompt(c, r.to_dict())))
        for k in ABL: jobs.append(("ablation", r.scenario_id, k, ablation_prompt(k, r.to_dict())))
    for _, r in se.iterrows():
        for c in ("R0", "R1", "R2"): jobs.append(("sensor", r.session_id, c, sensor_prompt(c, r.to_dict())))
    rng = random.Random(SEED); codes = set(); rows = []
    for n, (study, item, cond, p) in enumerate(jobs, 1):
        set_seed(SEED)
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(ids, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=260)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        words = text.split(); text = " ".join(words[:150])
        while True:
            code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
            if code not in codes: codes.add(code); break
        rows.append(dict(response_id=f"R{n:04d}", study=study, item_id=item, condition=cond, generator=model_name, generator_revision=revision,
                         temperature=0.7, top_p=0.9, max_words=150, seed=SEED, prompt_sha256=hashlib.sha256(p.encode()).hexdigest(),
                         response_text=text, word_count=len(text.split()), generated_utc=datetime.now(timezone.utc).isoformat(), blind_code=code))
    pd.DataFrame(rows).to_csv(d / "responses.csv", index=False)
    print(len(rows), "responses written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--revision", default=None)
    a = ap.parse_args(); main(a.data_dir, a.model, a.revision)
