import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from css.state import *
from css.stats import *

def test_expression_index_bounds():
    assert abs(expression_index({"fear": 1, **{e: 0 for e in EXPR_WEIGHTS if e != "fear"}}) - 100) < 1e-9
    assert expression_index({"happiness": 1, **{e: 0 for e in EXPR_WEIGHTS if e != "happiness"}}) == 0

def test_confidence_zero_when_uniform_or_unavailable():
    assert abs(confidence([1/3]*3, (1,1,1), (1,1,1))) < 1e-9
    assert confidence([0.9, 0.05, 0.05], (0,0,0), (1,1,1)) == 0

def test_trend():
    h = [StateRecord(0, 0.3, 0.8)]
    assert trend(h, StateRecord(10, 0.5, 0.8)) == "rising"
    assert trend(h, StateRecord(40, 0.5, 0.8)) is None      # reference too old
    assert trend([StateRecord(0, 0.3, 0.1)], StateRecord(5, 0.5, 0.8)) is None  # low confidence

def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]

def test_alpha_perfect_and_random():
    import numpy as np
    perfect = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]], float)
    assert abs(krippendorff_alpha_ordinal(perfect) - 1) < 1e-9
    rng = np.random.default_rng(1)
    rnd = rng.integers(1, 6, size=(3, 300)).astype(float)
    assert abs(krippendorff_alpha_ordinal(rnd)) < 0.1

def test_bootstrap_ci_contains_mean():
    m, lo, hi = paired_bootstrap_ci([0.5]*30 + [0.0]*30)
    assert lo < m < hi
