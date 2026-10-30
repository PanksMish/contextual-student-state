"""Priority-ordered rule selector (paper Section III). The selector is not evaluated in the paper;
only replies conditioned on the state are rated.

Order: explicit safety intent; elevated or rising stress; study planning; problem solving;
selected expressions; supportive check-in. States marked tentative (c_t < tau) never trigger the
stress rules, and interventions the student reported as unhelpful are skipped."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set
from .config import Thresholds
from .safety import detect_risk
from ..state import CSS

ESCALATE, GROUND, PLAN, SOLVE, ACKNOWLEDGE, CHECKIN = "escalate_safety", "grounding", "study_planning", "problem_solving", "acknowledge_emotion", "supportive_checkin"


@dataclass
class Decision:
    action: str
    reason: str


def select_action(css: CSS, utterance: str, intent: dict, th: Thresholds, failed: Optional[Set[str]] = None) -> Decision:
    failed = failed or set()
    if detect_risk(utterance)[0]:
        return Decision(ESCALATE, "explicit safety cue")
    s = css.current.s; trend = _trend(css)
    if not css.tentative() and GROUND not in failed and (s >= th.elevated_s or trend == "rising"):
        return Decision(GROUND, f"s={s:.2f}, trend={trend}")
    if PLAN not in failed and (intent.get("academic_stress", 0) > 0.5 or intent.get("exam_anxiety", 0) > 0.5):
        return Decision(PLAN, "academic intent")
    if SOLVE not in failed and intent.get("social_conflict", 0) > 0.5:
        return Decision(SOLVE, "interpersonal intent")
    if css.expression_index is not None and css.expression_index >= th.expression_alert and ACKNOWLEDGE not in failed:
        return Decision(ACKNOWLEDGE, f"expression index {css.expression_index:.0f}")
    return Decision(CHECKIN, "default")


def _trend(css: CSS) -> Optional[str]:
    from ..state import trend
    return trend(list(css.history), css.current)
