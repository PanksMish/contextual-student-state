"""Shared helpers: seeding, metrics, bootstrap, JSON output."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def macro_f1(y_true, y_pred, labels=None) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def cluster_bootstrap(values_by_cluster: dict, stat, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    """Resample clusters (students, conversations, subjects) with replacement.

    values_by_cluster: {cluster_id: (arrays...)}; stat receives the concatenated arrays.
    """
    rng = np.random.default_rng(seed)
    keys = list(values_by_cluster)
    est = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        parts = [values_by_cluster[keys[i]] for i in pick]
        cat = tuple(np.concatenate([p[j] for p in parts]) for j in range(len(parts[0])))
        est.append(stat(*cat))
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def save_json(obj, path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float))
