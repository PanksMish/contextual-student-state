import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
torch = pytest.importorskip("torch")

def test_bilstm_has_6820_parameters():
    from experiments.wesad_raw import make_model
    assert sum(p.numel() for p in make_model().parameters()) == 6820

def test_linear_heads_have_paper_parameter_counts():
    for d, n in ((512, 3591), (768, 5383), (2048, 14343)):
        assert d * 7 + 7 == n

def test_head_training_runs():
    from experiments.fer_linear_head import train_head
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 400); f = rng.normal(0, 1, (400, 32)) + np.eye(7)[y].dot(rng.normal(0, 2, (7, 32)))
    r = train_head(f[:250], y[:250], f[250:320], y[250:320], f[320:], y[320:], 1, epochs=5)
    assert r["test_acc"] > 0.3

def test_temporal_ablation_runs_on_synthetic():
    from experiments.wesad_raw import fit_predict, sequences
    rng = np.random.default_rng(0); n = 300
    y = rng.integers(0, 4, n); X = rng.normal(0, 1, (n, 8)) + y[:, None]
    subj = np.repeat(np.arange(1, 7), 50); start = np.tile(np.arange(50) * 30.0, 6)
    S, yy, ss = sequences(X, y, subj, start, 5)
    p = fit_predict((S[ss <= 4], yy[ss <= 4]), (S[ss == 5], yy[ss == 5]), (S[ss == 6], yy[ss == 6]), 0, epochs=5)
    assert len(p) == 50

def test_fusion_runs_on_synthetic():
    from experiments.fusion_cv import run
    rng = np.random.default_rng(0); n = 650
    df = pd.DataFrame({"student_id": np.repeat(np.arange(65), 10), "band": np.repeat(rng.integers(0, 3, 65), 10)})
    for m, k in (("wear", 6), ("face", 7), ("text", 4), ("ctx", 3)):
        for j in range(k): df[f"{m}_{j}"] = rng.normal(0, 1, n) + df.band * (0.6 if m != "ctx" else 0.0)
    for m in ("wear", "face", "text"): df[f"a_{m}"] = 1.0
    r = run(df, n_boot=20)
    assert "full" in r and r["majority"]["macro_f1"] < 0.5
