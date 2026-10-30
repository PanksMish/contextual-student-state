"""Statistics used for the human-rated evaluation."""
from __future__ import annotations
import numpy as np
from scipy import stats


def paired_bootstrap_ci(diff, n_boot=10000, seed=0, alpha=0.05):
    """Mean and percentile CI of per-scenario paired differences (resampling unit = scenario)."""
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def holm(pvals):
    """Holm-Bonferroni adjusted p values, in input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def wilcoxon_p(diff):
    d = np.asarray(diff, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def noninferior(diff, margin=0.30, n_boot=10000, seed=0, alpha_one_sided=0.025):
    """H4: lower one-sided bound of the mean difference must exceed -margin."""
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    lower = float(np.quantile(d[idx].mean(axis=1), alpha_one_sided))
    return lower, lower > -margin


def krippendorff_alpha_ordinal(ratings):
    """Ordinal Krippendorff alpha. `ratings`: raters x units array, NaN = missing."""
    r = np.asarray(ratings, float)
    vals = np.unique(r[~np.isnan(r)])
    k = len(vals)
    pos = {v: i for i, v in enumerate(vals)}
    o = np.zeros((k, k))
    for u in range(r.shape[1]):
        col = r[:, u]
        col = col[~np.isnan(col)]
        m = len(col)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[pos[col[i]], pos[col[j]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    cum = np.cumsum(n_c)
    def delta2(c, kk):
        lo, hi = min(c, kk), max(c, kk)
        s = n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2.0
        return s * s
    d_o = sum(o[c, kk] * delta2(c, kk) for c in range(k) for kk in range(k) if c != kk)
    d_e = sum(n_c[c] * n_c[kk] * delta2(c, kk) for c in range(k) for kk in range(k) if c != kk) / (n - 1)
    return float(1.0 - d_o / d_e) if d_e > 0 else float("nan")
