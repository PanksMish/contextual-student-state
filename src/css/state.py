"""Section III of the paper: confidence, expression index, missing observations, trend.

Every constant below is a prototype choice stated in the paper. None is fitted
to data, and c_t is a reliability heuristic, not a calibrated probability.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

TAU = 0.3            # confidence threshold; states below it are marked tentative
MAX_AGE_MIN = 30.0   # oldest admissible reference state for the change
TREND_EPS = 0.1      # |delta s| above this is rising or falling
MEMORY_K = 10        # bounded memory of recent states

EXPR_WEIGHTS = {     # hand-set before any experiment, not fitted
    "anger": 0.8, "disgust": 0.6, "fear": 1.0, "happiness": 0.0,
    "neutral": 0.1, "sadness": 0.7, "surprise": 0.3,
}


def expression_index(probs: Dict[str, float]) -> float:
    """E_t = 100 * sum_e w_e p_e, in [0, 100] because every w_e is in [0, 1]."""
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("expression probabilities must sum to a positive value")
    return 100.0 * sum(EXPR_WEIGHTS[e] * p / total for e, p in probs.items())


def stress_estimate(p_low: float, p_mod: float, p_high: float) -> float:
    """Expected severity s_t = 0.5 p_mod + p_high, in [0, 1]."""
    return 0.5 * p_mod + p_high


def entropy(p: Sequence[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def quality_wearable(flagged_fraction: float) -> float:
    """One minus the share of the window flagged for motion or contact loss."""
    return min(1.0, max(0.0, 1.0 - flagged_fraction))


def quality_face(detector_score: float, sharpness_size: float) -> float:
    return min(1.0, max(0.0, detector_score * sharpness_size))


def quality_audio(snr_db: float) -> float:
    return min(1.0, max(0.0, snr_db / 20.0))


def confidence(p: Sequence[float], available: Sequence[int], quality: Sequence[float]) -> float:
    """c_t = (1 - H(p)/log K) * mean_m(a_m q_m)."""
    k = len(p)
    sharp = 1.0 - entropy(p) / math.log(k)
    rho = sum(a * q for a, q in zip(available, quality)) / len(available)
    return sharp * rho


@dataclass
class StateRecord:
    t_min: float                      # minutes since session start
    s: float
    c: float
    available: Sequence[int] = (1, 1, 1)
    quality: Sequence[float] = (1.0, 1.0, 1.0)


def trend(history: List[StateRecord], current: StateRecord) -> Optional[str]:
    """Rising / falling / stable, or None (unknown) if no admissible reference."""
    ref = None
    for rec in reversed(history):
        if rec.c >= TAU and (current.t_min - rec.t_min) <= MAX_AGE_MIN and rec.t_min < current.t_min:
            ref = rec
            break
    if ref is None:
        return None
    d = current.s - ref.s
    if d > TREND_EPS:
        return "rising"
    if d < -TREND_EPS:
        return "falling"
    return "stable"


@dataclass
class CSS:
    """Contextual Student State rendered into a prompt (see conditions.py)."""
    current: StateRecord
    expression_index: Optional[float]
    academic_context: str
    dialogue_context: str
    interventions: List[Dict[str, str]] = field(default_factory=list)
    history: List[StateRecord] = field(default_factory=list)

    def memory(self) -> List[StateRecord]:
        return self.history[-MEMORY_K:]

    def tentative(self) -> bool:
        return self.current.c < TAU

    def missing(self) -> List[str]:
        names = ("wearable", "face", "voice")
        return [n for n, a in zip(names, self.current.available) if not a]
