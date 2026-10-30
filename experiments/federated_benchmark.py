"""Five-seed federated benchmark (Section V-E).

A linear 8->4 softmax head (36 parameters, 144 bytes per float32 update) is trained either
centrally on the pooled training windows or with FedAvg over three clients. Both start from
identical parameters and use cross-entropy, SGD (lr 0.05, momentum 0.9), batch size 256, and
15 comparison steps. In FedAvg a step is a round: each client sees its own examples once, then
the server averages parameters weighted by client sample counts. The centralised step is one
pass over the pooled training windows. The step with the best validation macro F1 is selected
and the test set is used afterwards.

Input: an .npz with X (windows x 8 features), y (0..3), subject (int). Windows are 12-row
mean-pools of the public HR/HRV feature mirror (see build_windows). Subject-disjoint splits and
clients default to the paper (train 2-10, val 11/14/15, test 13/16/17; clients 2/5/8, 3/6/9, 4/7/10).

usage: python -m experiments.federated_benchmark DATA.npz OUT.json
"""
from __future__ import annotations
import argparse
import numpy as np
from .common import macro_f1, save_json

SEEDS = (20260921, 20260922, 20260923, 20260924, 20260925)
TRAIN = tuple(range(2, 11)); VAL = (11, 14, 15); TEST = (13, 16, 17)
CLIENTS = ((2, 5, 8), (3, 6, 9), (4, 7, 10))
FEATURES = ("HR", "RMSSD", "SDRR", "VLF", "LF", "HF", "LF/HF", "SD1")


def build_windows(rows: np.ndarray, labels: np.ndarray, width: int = 12):
    """Mean-pool non-overlapping groups of `width` rows; label = majority label of the group."""
    n = len(rows) // width
    X = rows[: n * width].reshape(n, width, -1).mean(1)
    L = labels[: n * width].reshape(n, width)
    y = np.array([np.bincount(r.astype(int)).argmax() for r in L])
    return X, y


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Linear:
    def __init__(self, d, k, rng):
        self.W = rng.normal(0, 0.01, (d, k)); self.b = np.zeros(k)

    def params(self): return np.concatenate([self.W.ravel(), self.b])

    def set(self, v):
        d, k = self.W.shape
        self.W = v[: d * k].reshape(d, k).copy(); self.b = v[d * k:].copy()

    def predict(self, X): return softmax(X @ self.W + self.b).argmax(1)


def sgd_epoch(model, X, y, lr, mom, batch, rng, buf=None):
    n, k = len(X), model.W.shape[1]
    buf = np.zeros_like(model.params()) if buf is None else buf
    for i in range(0, n, batch):
        idx = np.arange(n)[i:i + batch]
        p = softmax(X[idx] @ model.W + model.b)
        p[np.arange(len(idx)), y[idx]] -= 1.0
        gW = X[idx].T @ p / len(idx); gb = p.mean(0)
        g = np.concatenate([gW.ravel(), gb])
        buf = mom * buf + g
        model.set(model.params() - lr * buf)
    return buf


def run_seed(X, y, subj, seed, steps=15, lr=0.05, mom=0.9, batch=256):
    rng = np.random.default_rng(seed)
    tr, va, te = (np.isin(subj, s) for s in (TRAIN, VAL, TEST))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Z = (X - mu) / sd
    init = Linear(Z.shape[1], 4, rng).params()
    out = {}
    for method in ("central", "fedavg"):
        m = Linear(Z.shape[1], 4, rng); m.set(init.copy())
        rows = []
        # shuffle once per seed so batches are not label-sorted
        order = np.random.default_rng(seed).permutation(tr.sum())
        Xc, yc = Z[tr][order], y[tr][order]
        clients = [(Z[np.isin(subj, c)], y[np.isin(subj, c)]) for c in CLIENTS]
        clients = [(a[np.random.default_rng(seed + i).permutation(len(a))], b[np.random.default_rng(seed + i).permutation(len(b))])
                   for i, (a, b) in enumerate(clients)]
        buf = None
        for step in range(1, steps + 1):
            if method == "central":
                buf = sgd_epoch(m, Xc, yc, lr, mom, batch, rng, buf)
            else:
                global_v = m.params(); updates, weights = [], []
                for Xk, yk in clients:
                    loc = Linear(Z.shape[1], 4, rng); loc.set(global_v.copy())
                    sgd_epoch(loc, Xk, yk, lr, mom, batch, rng, None)
                    updates.append(loc.params()); weights.append(len(Xk))
                m.set(np.average(updates, axis=0, weights=weights))
            rows.append(dict(step=step, val=macro_f1(y[va], m.predict(Z[va])), test=macro_f1(y[te], m.predict(Z[te]))))
        best = max(rows, key=lambda r: r["val"])
        out[method] = dict(selected_step=best["step"], test_macro_f1=best["test"], val_macro_f1=best["val"], curve=rows)
    out["diff_fedavg_minus_central"] = out["fedavg"]["test_macro_f1"] - out["central"]["test_macro_f1"]
    return out


def communication(n_params=36, rounds=15, clients=3, n_train_windows=10317, n_features=8):
    upd = n_params * 4
    # one upload and one download per client and round; raw upload = float32 features only
    return dict(update_bytes=upd, total_fed_bytes=upd * rounds * clients * 2,
                upload_all_windows_bytes=n_train_windows * n_features * 4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out")
    a = ap.parse_args()
    z = np.load(a.data); X, y, subj = z["X"], z["y"].astype(int), z["subject"]
    res = {s: run_seed(X, y, subj, s) for s in SEEDS}
    c = [r["central"]["test_macro_f1"] for r in res.values()]; f = [r["fedavg"]["test_macro_f1"] for r in res.values()]
    summary = dict(central_mean=float(np.mean(c)), central_sd=float(np.std(c, ddof=1)),
                   fedavg_mean=float(np.mean(f)), fedavg_sd=float(np.std(f, ddof=1)),
                   paired_diffs=[r["diff_fedavg_minus_central"] for r in res.values()], communication=communication())
    save_json(dict(per_seed=res, summary=summary), a.out)
    print(summary)
