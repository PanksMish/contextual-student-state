#!/usr/bin/env python3
"""Reproduce the human-rated tables from scenarios/responses/ratings CSVs.

usage: python scripts/analyze_ratings.py DATA_DIR OUT_DIR
Writes OUT_DIR/main_table.csv, sensor_table.csv, alpha.csv, report.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from css.stats import (paired_bootstrap_ci, holm, wilcoxon_p, noninferior,
                       krippendorff_alpha_ordinal)

SCALES = ["relevance", "appropriateness", "personalisation", "empathy", "safety"]
FLAGS = ["flag_diagnostic_or_medication", "flag_unsafe_advice", "flag_missed_explicit_risk",
         "flag_inappropriate_reassurance", "flag_needs_human_review"]


def load(d):
    d = Path(d)
    return (pd.read_csv(d / "responses.csv"), pd.read_csv(d / "ratings.csv"))


def item_means(resp, rat, study, conds):
    r = resp[(resp.study == study) & resp.condition.isin(conds)]
    m = rat.merge(r[["response_id", "item_id", "condition"]], on="response_id")
    return m.groupby(["item_id", "condition"])[SCALES].mean().reset_index()


def contrast(table, a, b, scale):
    wide = table.pivot(index="item_id", columns="condition", values=scale)
    return (wide[a] - wide[b]).dropna().to_numpy()


def main(data, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    resp, rat = load(data)
    rep = {}
    # ---- main pilot
    t = item_means(resp, rat, "pilot", ["C0", "C1", "CF", "C2"])
    rows = []
    for s in SCALES:
        w = t.pivot(index="item_id", columns="condition", values=s)
        d = contrast(t, "C2", "C1", s)
        mean, lo, hi = paired_bootstrap_ci(d)
        rows.append(dict(outcome=s, C0=w.C0.mean(), C1=w.C1.mean(), CF=w.CF.mean(), C2=w.C2.mean(),
                         diff_C2_C1=mean, ci_lo=lo, ci_hi=hi, raw_p=wilcoxon_p(d)))
    tab = pd.DataFrame(rows)
    hol = [i for i, s in enumerate(tab.outcome) if s != "safety"]
    adj = holm(tab.loc[hol, "raw_p"])
    tab["holm_p"] = np.nan
    tab.loc[hol, "holm_p"] = adj
    tab.to_csv(out / "main_table.csv", index=False)
    # extra contrasts
    for name, (a, b) in {"H3_C1_minus_C0": ("C1", "C0"), "H5_C2_minus_CF": ("C2", "CF")}.items():
        rep[name] = {s: paired_bootstrap_ci(contrast(t, a, b, s)) for s in ("relevance", "appropriateness")}
    low, ok = noninferior(contrast(t, "C2", "C1", "safety"))
    rep["H4_safety_lower_bound"] = low; rep["H4_noninferior_at_0.30"] = bool(ok)
    # alpha per scale (ordinal), raters x responses
    al = []
    for s in SCALES:
        pv = rat[rat.response_id.isin(resp[resp.study == "pilot"].response_id)].pivot(
            index="rater_id", columns="response_id", values=s)
        al.append(dict(scale=s, alpha=krippendorff_alpha_ordinal(pv.to_numpy())))
    pd.DataFrame(al).to_csv(out / "alpha.csv", index=False)
    # flags per condition
    fl = rat.merge(resp[["response_id", "condition", "study"]], on="response_id")
    fl = fl[fl.study == "pilot"]
    rep["review_flags_by_condition"] = (fl.groupby(["condition", "response_id"])["flag_needs_human_review"].max()
                                          .groupby("condition").sum().to_dict())
    # ---- sensor study
    st = item_means(resp, rat, "sensor", ["R0", "R1", "R2"])
    if len(st):
        sess = pd.read_csv(Path(data) / "sensor_sessions.csv").set_index("session_id")
        rows = []
        for s in SCALES[:2] + ["safety"]:
            w = st.pivot(index="item_id", columns="condition", values=s)
            d10 = (w.R1 - w.R0).dropna(); d21 = (w.R2 - w.R1).dropna()
            rows.append(dict(outcome=s, R0=w.R0.mean(), R1=w.R1.mean(), R2=w.R2.mean(),
                             R1_minus_R0=paired_bootstrap_ci(d10)[0], R1_R0_lo=paired_bootstrap_ci(d10)[1],
                             R1_R0_hi=paired_bootstrap_ci(d10)[2], R2_minus_R1=paired_bootstrap_ci(d21)[0],
                             R2_R1_lo=paired_bootstrap_ci(d21)[1], R2_R1_hi=paired_bootstrap_ci(d21)[2]))
            if s == "relevance":
                match = sess.loc[d10.index, "predicted_matches_reference"].astype(bool)
                rep["sensor_relevance_gain_predicted_correct"] = float(d10[match.values].mean())
                rep["sensor_relevance_gain_predicted_wrong"] = float(d10[~match.values].mean())
                rep["sensor_n_correct"] = int(match.sum()); rep["sensor_n_wrong"] = int((~match).sum())
                rep["weighted_check"] = float((match.sum()*d10[match.values].mean()+(~match).sum()*d10[~match.values].mean())/len(d10))
        pd.DataFrame(rows).to_csv(out / "sensor_table.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(tab.round(3).to_string(index=False))
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
