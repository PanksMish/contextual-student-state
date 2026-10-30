"""WESAD from raw chest ECG, with a five-window temporal ablation (Section V-F).

Pipeline
  1. Load SX.pkl (chest ECG at 700 Hz, label stream at 700 Hz). Keep labels 1 baseline,
     2 stress, 3 amusement, 4 meditation.
  2. Band-pass the ECG (5-15 Hz), detect R peaks, build the RR series.
  3. Cut windows (default 60 s, step 30 s, label purity >= 0.9) and compute eight features:
     HR, RMSSD, SDRR, VLF, LF, HF, LF/HF, SD1 (spectral bands from a 4 Hz resampled RR series).
  4. Temporal model: BiLSTM(8 -> 24, bidirectional) + LayerNorm(48) + Linear(48 -> 4) on the last
     five windows (6,820 parameters). Baseline: the same network on the current window only.
  5. Subject-disjoint evaluation (leave-one-subject-out; two training subjects held out for early
     stopping). The statistic is the per-subject macro F1 difference (temporal minus current).

Window length, step and purity are prototype choices (FIX in the paper if they differ).

usage:
  python -m experiments.wesad_raw features WESAD_DIR OUT.npz
  python -m experiments.wesad_raw ablate OUT.npz RESULT.json
"""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
from scipy import signal, interpolate
from .common import set_seed, macro_f1, save_json

FS = 700
LABELS = {1: 0, 2: 1, 3: 2, 4: 3}


def r_peaks(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    b, a = signal.butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    f = signal.filtfilt(b, a, ecg)
    e = np.convolve(np.gradient(f) ** 2, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    thr = np.percentile(e, 90) * 0.3
    peaks, _ = signal.find_peaks(e, height=thr, distance=int(0.3 * fs))
    # refine to the local maximum of the filtered ECG
    w = int(0.05 * fs)
    return np.array([max(0, p - w) + int(np.argmax(f[max(0, p - w):p + w])) for p in peaks])


def hrv_features(rr_s: np.ndarray, t_s: np.ndarray) -> np.ndarray | None:
    """Eight features from RR intervals (seconds) with their time stamps."""
    if len(rr_s) < 20:
        return None
    ok = (rr_s > 0.3) & (rr_s < 2.0)
    rr, t = rr_s[ok], t_s[ok]
    if len(rr) < 20:
        return None
    hr = 60.0 / rr.mean(); sdrr = rr.std(ddof=1) * 1000
    d = np.diff(rr) * 1000; rmssd = np.sqrt(np.mean(d ** 2)); sd1 = np.std(d, ddof=1) / np.sqrt(2)
    grid = np.arange(t[0], t[-1], 0.25)
    x = interpolate.interp1d(t, rr * 1000, kind="cubic", fill_value="extrapolate")(grid)
    x = x - x.mean()
    f, p = signal.welch(x, fs=4.0, nperseg=min(len(x), 256))
    band = lambda lo, hi: float(np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]))
    vlf, lf, hf = band(0.003, 0.04), band(0.04, 0.15), band(0.15, 0.4)
    return np.array([hr, rmssd, sdrr, vlf, lf, hf, lf / (hf + 1e-9), sd1])


def subject_windows(ecg, labels, win_s=60, step_s=30, purity=0.9):
    pk = r_peaks(ecg); tp = pk / FS; rr = np.diff(tp); tr = tp[1:]
    X, y, start = [], [], []
    for s in np.arange(0, len(ecg) / FS - win_s, step_s):
        lab = labels[int(s * FS):int((s + win_s) * FS)]
        lab = lab[np.isin(lab, list(LABELS))]
        if len(lab) < purity * win_s * FS:
            continue
        top = np.bincount(lab).argmax()
        if (lab == top).mean() < purity:
            continue
        m = (tr >= s) & (tr < s + win_s)
        ft = hrv_features(rr[m], tr[m])
        if ft is None:
            continue
        X.append(ft); y.append(LABELS[int(top)]); start.append(s)
    return np.array(X), np.array(y), np.array(start)


def build_features(wesad_dir: str, out: str):
    Xs, ys, ss, ts = [], [], [], []
    for p in sorted(Path(wesad_dir).glob("S*/S*.pkl")):
        d = pickle.load(open(p, "rb"), encoding="latin1")
        sid = int(p.stem[1:])
        X, y, t = subject_windows(d["signal"]["chest"]["ECG"].ravel(), d["label"].ravel())
        Xs.append(X); ys.append(y); ss.append(np.full(len(y), sid)); ts.append(t)
    np.savez(out, X=np.concatenate(Xs), y=np.concatenate(ys), subject=np.concatenate(ss), start=np.concatenate(ts))


# ---------- temporal ablation (needs torch) ----------
def make_model():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(8, 24, batch_first=True, bidirectional=True)
            self.norm = nn.LayerNorm(48); self.head = nn.Linear(48, 4)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.head(self.norm(h[:, -1]))
    return Net()


def sequences(X, y, subj, start, length):
    """Last `length` consecutive windows (step-aligned) of the same subject; padded at the start by repetition."""
    Xs, ys, ss = [], [], []
    for sid in np.unique(subj):
        idx = np.where(subj == sid)[0]; idx = idx[np.argsort(start[idx])]
        for k, i in enumerate(idx):
            sel = idx[max(0, k - length + 1):k + 1]
            seq = X[sel]
            if len(seq) < length:
                seq = np.vstack([np.repeat(seq[:1], length - len(seq), 0), seq])
            Xs.append(seq); ys.append(y[i]); ss.append(sid)
    return np.array(Xs), np.array(ys), np.array(ss)


def fit_predict(train, val, test, seed, epochs=60, lr=3e-3):
    import torch
    set_seed(seed)
    (Xa, ya), (Xv, yv), (Xt, _) = train, val, test
    mu, sd = Xa.reshape(-1, 8).mean(0), Xa.reshape(-1, 8).std(0) + 1e-8
    f = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    m = make_model(); opt = torch.optim.Adam(m.parameters(), lr=lr)
    cnt = np.bincount(ya, minlength=4).astype(float)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), dtype=torch.float32))
    XA, YA = f(Xa), torch.tensor(ya)
    best, state = -1, None
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(XA))
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]; opt.zero_grad(); lossf(m(XA[j]), YA[j]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            s = macro_f1(yv, m(f(Xv)).argmax(1).numpy())
        if s > best:
            best, state = s, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return m(f(Xt)).argmax(1).numpy()


def ablate(npz: str, out: str, seed=20260915):
    z = np.load(npz); X, y, subj, start = z["X"], z["y"].astype(int), z["subject"], z["start"]
    S5, y5, s5 = sequences(X, y, subj, start, 5)
    S1, y1, s1 = sequences(X, y, subj, start, 1)
    subs = np.unique(subj); rows = []
    for k, test_sub in enumerate(subs):
        rest = [s for s in subs if s != test_sub]; val_sub = rest[k % len(rest)], rest[(k + 7) % len(rest)]
        tr = ~np.isin(s5, [test_sub, *val_sub]); va = np.isin(s5, val_sub); te = s5 == test_sub
        p5 = fit_predict((S5[tr], y5[tr]), (S5[va], y5[va]), (S5[te], y5[te]), seed)
        p1 = fit_predict((S1[tr], y1[tr]), (S1[va], y1[va]), (S1[te], y1[te]), seed)
        rows.append(dict(subject=int(test_sub), f1_temporal=macro_f1(y5[te], p5), f1_current=macro_f1(y1[te], p1)))
    d = np.array([r["f1_temporal"] - r["f1_current"] for r in rows])
    res = dict(per_subject=rows, mean_temporal=float(np.mean([r["f1_temporal"] for r in rows])),
               mean_current=float(np.mean([r["f1_current"] for r in rows])), mean_diff=float(d.mean()),
               subjects_improved=int((d > 0).sum()), n_subjects=len(d))
    save_json(res, out); print(res["mean_temporal"], res["mean_current"], res["subjects_improved"], "/", len(d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features"); f.add_argument("wesad_dir"); f.add_argument("out")
    g = sub.add_parser("ablate"); g.add_argument("npz"); g.add_argument("out")
    a = ap.parse_args()
    build_features(a.wesad_dir, a.out) if a.cmd == "features" else ablate(a.npz, a.out)
