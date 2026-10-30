import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd

def test_communication_matches_paper():
    from experiments.federated_benchmark import communication
    c = communication()
    assert c["update_bytes"] == 144 and c["total_fed_bytes"] == 12960 and c["upload_all_windows_bytes"] == 330144

def test_federated_runs_on_synthetic():
    from experiments.federated_benchmark import run_seed
    rng = np.random.default_rng(0)
    subj = np.repeat(np.arange(2, 18), 60); y = rng.integers(0, 4, len(subj))
    X = rng.normal(0, 1, (len(subj), 8)) + y[:, None] * 0.8
    r = run_seed(X, y, subj, 20260921)
    assert 1 <= r["central"]["selected_step"] <= 15 and r["fedavg"]["test_macro_f1"] > 0.4

def test_wer_and_normalisation():
    from experiments.asr_wer import wer, normalise
    assert wer("the cat sat", "the cat sat") == (0, 3)
    assert wer("the cat sat", "the bat") == (2, 3)
    assert normalise("Hello,  World!") == "hello world"
    assert wer("मैं परीक्षा से डरता हूँ", "मैं परीक्षा से डरता हूँ") == (0, 5)

def test_wearable_validation():
    from experiments.wearable_validation import summarise
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"participant": np.repeat(np.arange(20), 12)})
    df["reference"] = rng.normal(80, 8, len(df)); df["device"] = df.reference + rng.normal(0.1, 2.0, len(df))
    r = summarise(df, n_boot=300)
    assert r["n_pairs"] == 240 and r["n_participants"] == 20 and r["mae_ci"][0] < r["mae"] < r["mae_ci"][1]
    assert r["loa"][0] < r["bias"] < r["loa"][1]

def test_pixel_disjoint_split_rules():
    from experiments.fer_splits import pixel_disjoint
    rows = []
    for i in range(300):
        lab = i % 7; h = f"h{i}"
        rows.append(dict(path=f"train/{lab}/{i}.png", split="train", label=str(lab), sha_pixels=h))
    for i in range(280, 330):   # 20 test files duplicate train hashes, 30 are new
        lab = i % 7; rows.append(dict(path=f"test/{lab}/{i}.png", split="test", label=str(lab), sha_pixels=f"h{i}"))
    rows += [dict(path="train/1/x.png", split="train", label="1", sha_pixels="c1"), dict(path="val/2/y.png", split="val", label="2", sha_pixels="c1")]
    out = pixel_disjoint(pd.DataFrame(rows))
    assert "c1" not in set(out.sha_pixels)                      # conflicting labels dropped
    assert out.sha_pixels.is_unique                              # one image per group
    tst = set(out[out.assigned == "test"].sha_pixels); rest = set(out[out.assigned != "test"].sha_pixels)
    assert not tst & rest                                        # no pixel group crosses the boundary

def test_hrv_and_rpeaks_on_synthetic_ecg():
    from experiments.wesad_raw import r_peaks, hrv_features, FS
    hr = 72.0; t = np.arange(0, 60, 1 / FS); ecg = np.zeros_like(t)
    beat = np.arange(0.5, 60, 60 / hr)
    for b in beat: ecg += np.exp(-((t - b) ** 2) / (2 * 0.01 ** 2))
    ecg += np.random.default_rng(0).normal(0, 0.02, len(t))
    pk = r_peaks(ecg); tp = pk / FS
    ft = hrv_features(np.diff(tp), tp[1:])
    assert abs(ft[0] - hr) < 1.5 and len(ft) == 8
