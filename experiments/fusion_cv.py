"""Multimodal fusion with student-disjoint cross-validation (Sections V-B and VIII-C).

Input: windows.csv with one row per 10-s window and columns
  student_id, band (0 low, 1 moderate, 2 high),
  wear_*   window summary of the wearable BiLSTM (trained inside each fold in the paper; a
           precomputed per-fold summary column set `wear_*` is accepted here),
  face_*   seven expression probabilities,
  text_*   four intent probabilities,
  ctx_*    context vector,
  q_wear, q_face, q_text   quality in [0,1];  a_wear, a_face, a_text availability in {0,1}.

Model: MLP over [features, availability mask] with one hidden layer of 64 units, dropout 0.3,
trained with modality dropout (p=0.3 per modality, at least one kept). Five student-disjoint
folds; scores are pooled over folds; bootstrap intervals resample students.
Reports full model, majority baseline, single-modality models, leave-one-modality-out at test
time, and 10 dB Gaussian noise on the wearable and face streams.

usage: python -m experiments.fusion_cv windows.csv OUT.json [--seed 20260918]
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from .common import set_seed, macro_f1, cluster_bootstrap, save_json

MODS = ("wear", "face", "text")


def columns(df):
    return {m: [c for c in df.columns if c.startswith(m + "_")] for m in MODS} | {"ctx": [c for c in df.columns if c.startswith("ctx_")]}


def assemble(df, cols, use=MODS, noise_db=None, rng=None, drop=()):
    parts, avail = [], []
    for m in MODS:
        x = df[cols[m]].to_numpy(float)
        a = df[f"a_{m}"].to_numpy(float) if m in use and m not in drop else np.zeros(len(df))
        if noise_db is not None and m in noise_db and a.any():
            p = np.mean(x ** 2) + 1e-12
            x = x + rng.normal(0, np.sqrt(p / 10 ** (noise_db[m] / 10)), x.shape)
        parts.append(x * a[:, None]); avail.append(a)
    parts.append(df[cols["ctx"]].to_numpy(float))
    return np.concatenate(parts + [np.stack(avail, 1)], 1), np.stack(avail, 1)


def fit(Xtr, Atr, ytr, seed, epochs=40, nmods=3):
    import torch
    set_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(64, 3))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    X, y = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    k = nmods; d_mask = Xtr.shape[1] - k
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(X))
        for i in range(0, len(perm), 128):
            j = perm[i:i + 128]; xb = X[j].clone()
            keep = (torch.rand(len(j), k) > 0.3).float()
            none = keep.sum(1) == 0
            keep[none, torch.randint(0, k, (int(none.sum()),))] = 1.0
            # zero the dropped modality's features and its mask entry; feature slices are equal-sized blocks recorded in `blocks`
            for mi, (a, b) in enumerate(fit.blocks):
                xb[:, a:b] *= keep[:, mi:mi + 1]
            xb[:, d_mask:] *= keep
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(xb), y[j]).backward(); opt.step()
    net.eval(); return net


def run(df, seed=20260918, n_boot=2000):
    import torch
    cols = columns(df); y = df.band.to_numpy(int); g = df.student_id.to_numpy()
    sizes = [len(cols[m]) for m in MODS]; off = np.cumsum([0] + sizes)
    fit.blocks = [(int(off[i]), int(off[i + 1])) for i in range(3)]
    configs = {"full": dict(), "wearable_only": dict(use=("wear",)), "face_only": dict(use=("face",)), "text_only": dict(use=("text",)),
               "no_wearable": dict(drop=("wear",)), "no_face": dict(drop=("face",)), "no_text": dict(drop=("text",)),
               "noise10db_wear": dict(noise_db={"wear": 10}), "noise10db_face": dict(noise_db={"face": 10})}
    preds = {k: np.zeros(len(df), int) for k in configs}
    rng = np.random.default_rng(seed)
    for fold, (tr, te) in enumerate(GroupKFold(5).split(df, y, g)):
        Xtr, Atr = assemble(df.iloc[tr], cols)
        net_full = fit(Xtr, Atr, y[tr], seed + fold)
        singles = {k: fit(*assemble(df.iloc[tr], cols, use=cfg["use"]), y[tr], seed + fold)
                   for k, cfg in configs.items() if "use" in cfg}
        for k, cfg in configs.items():
            net = singles.get(k, net_full)
            Xte, _ = assemble(df.iloc[te], cols, rng=rng, **cfg)
            with torch.no_grad():
                preds[k][te] = net(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
    preds["majority"] = np.full(len(df), np.bincount(y).argmax())
    res = {}
    for k, p in preds.items():
        by = {s: (y[g == s], p[g == s]) for s in np.unique(g)}
        res[k] = dict(macro_f1=macro_f1(y, p), ci=cluster_bootstrap(by, macro_f1, n_boot, seed))
    return res


def train_final(df, out, seed=20260918):
    """Train on all students and save the weights for src/css/runtime/fusion.py."""
    import torch
    cols = columns(df); y = df.band.to_numpy(int)
    sizes = {m: len(cols[m]) for m in MODS} | {"ctx": len(cols["ctx"])}
    off = np.cumsum([0] + [sizes[m] for m in MODS]); fit.blocks = [(int(off[i]), int(off[i + 1])) for i in range(3)]
    X, A = assemble(df, cols); net = fit(X, A, y, seed)
    torch.save({"state_dict": net.state_dict(), "sizes": sizes}, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("windows"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=20260918); ap.add_argument("--save-model")
    a = ap.parse_args(); d = pd.read_csv(a.windows); r = run(d, a.seed); save_json(r, a.out); print(r)
    if a.save_model: train_final(d, a.save_model, a.seed)
