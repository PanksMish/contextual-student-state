"""Wearable against reference devices (Supplement S4).

Input CSV: participant, measure (hr|spo2|temp), device, reference.
For each measure: MAE with a participant-level bootstrap (resample participants, 5,000 draws),
Bland-Altman bias and 95% limits of agreement (bias +- 1.96 SD of differences), and an optional
HR agreement figure.

usage: python -m experiments.wearable_validation PAIRS.csv OUT.json [--fig bland_altman_hr.png]
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from .common import cluster_bootstrap, save_json


def summarise(df: pd.DataFrame, n_boot=5000, seed=0):
    d = (df.device - df.reference).to_numpy()
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    clusters = {p: (np.abs((g.device - g.reference).to_numpy()),) for p, g in df.groupby("participant")}
    lo, hi = cluster_bootstrap(clusters, lambda a: a.mean(), n_boot, seed)
    return dict(n_pairs=len(df), n_participants=df.participant.nunique(), mae=float(np.abs(d).mean()),
                mae_ci=[lo, hi], bias=bias, loa=[bias - 1.96 * sd, bias + 1.96 * sd])


def bland_altman_figure(df, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = (df.device + df.reference) / 2; d = df.device - df.reference
    b, s = d.mean(), d.std(ddof=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(m, d, s=8, alpha=0.6)
    ax.axhline(b, color="k"); ax.axhline(b + 1.96 * s, color="k", ls="--"); ax.axhline(b - 1.96 * s, color="k", ls="--")
    ax.set_xlabel("Mean of wearable and reference heart rate (bpm)"); ax.set_ylabel("Wearable minus reference (bpm)")
    fig.tight_layout(); fig.savefig(path, dpi=300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pairs"); ap.add_argument("out"); ap.add_argument("--fig")
    a = ap.parse_args(); df = pd.read_csv(a.pairs)
    res = {m: summarise(g) for m, g in df.groupby("measure")}
    save_json(res, a.out); print(res)
    if a.fig and "hr" in set(df.measure):
        bland_altman_figure(df[df.measure == "hr"], a.fig)
