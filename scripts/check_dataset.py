#!/usr/bin/env python3
"""Check a dataset directory against the design counts of the paper (docs/HUMAN_RATED_DATASET_SPEC.md)."""
import sys
from pathlib import Path
import pandas as pd

d = Path(sys.argv[1]); errs = []
resp = pd.read_csv(d / "responses.csv"); rat = pd.read_csv(d / "ratings.csv")
exp = {"pilot": 240, "ablation": 180, "sensor": 120}
for k, v in exp.items():
    n = int((resp.study == k).sum())
    if n != v: errs.append(f"{k}: {n} responses, expected {v}")
if len(rat) != 3 * len(resp): errs.append(f"ratings {len(rat)} != 3 x responses {len(resp)}")
for c in ["relevance", "appropriateness", "personalisation", "empathy", "safety"]:
    if rat[c].isna().any() or not rat[c].between(1, 5).all(): errs.append(f"{c}: missing or out of 1-5")
if (d / "scenarios.csv").exists():
    sc = pd.read_csv(d / "scenarios.csv")
    if len(sc) != 60: errs.append(f"scenarios: {len(sc)}, expected 60")
    if "frozen_utc" in sc and "generated_utc" in resp:
        if pd.to_datetime(sc.frozen_utc).max() >= pd.to_datetime(resp.generated_utc).min():
            errs.append("scenarios were not frozen before the first response was generated")
    if (pd.to_datetime(resp.get("generated_utc", pd.Series(dtype=str))) < pd.Timestamp("2026-01-01", tz="UTC")).any():
        pass
print("OK" if not errs else "\n".join(errs)); sys.exit(1 if errs else 0)
