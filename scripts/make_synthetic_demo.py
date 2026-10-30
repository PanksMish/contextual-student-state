#!/usr/bin/env python3
"""Write a SYNTHETIC dataset with the right shape to smoke-test the pipeline.
The values are random. Never report them. Output: data/synthetic_demo/ (git-ignored by convention in the paper)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_demo"); out.mkdir(parents=True, exist_ok=True)
SC = ["relevance", "appropriateness", "personalisation", "empathy", "safety"]
FL = ["flag_diagnostic_or_medication", "flag_unsafe_advice", "flag_missed_explicit_risk",
      "flag_inappropriate_reassurance", "flag_needs_human_review"]
resp, items = [], []
def add(study, item, cond, mu):
    resp.append(dict(response_id=f"R{len(resp)+1:04d}", study=study, item_id=item, condition=cond)); items.append(mu)
for i in range(60):
    sid = f"S{i+1:03d}"
    for c, mu in zip(["C0", "C1", "CF", "C2"], [3.2, 3.4, 3.8, 4.0]): add("pilot", sid, c, mu)
    for c, mu in zip(["no_history", "no_academic", "no_interv"], [3.6, 3.7, 3.8]): add("ablation", sid, c, mu)
sess = []
for i in range(40):
    pid = f"P{i+1:03d}"; ok = int(i < 27)
    sess.append(dict(session_id=pid, predicted_matches_reference=ok))
    for c, mu in zip(["R0", "R1", "R2"], [3.2, 3.5 if ok else 3.1, 3.9]): add("sensor", pid, c, mu)
resp = pd.DataFrame(resp); resp["mu"] = items
rows = []
for _, r in resp.iterrows():
    base = rng.normal(0, 0.4)
    for rater in ("rater_1", "rater_2", "rater_3"):
        sc = {s: int(np.clip(round(r.mu + base + rng.normal(0, 0.5) + (0.9 if s == "safety" else 0)), 1, 5)) for s in SC}
        rows.append(dict(rating_id=f"RT{len(rows)+1:05d}", response_id=r.response_id, rater_id=rater, session=1, **sc,
                         **{f: int(rng.random() < 0.03) for f in FL}))
resp.drop(columns="mu").to_csv(out / "responses.csv", index=False)
pd.DataFrame(rows).to_csv(out / "ratings.csv", index=False)
pd.DataFrame(sess).to_csv(out / "sensor_sessions.csv", index=False)
print("wrote synthetic demo to", out, "(random values; not results)")
