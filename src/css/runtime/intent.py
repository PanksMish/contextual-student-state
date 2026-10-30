"""Runtime intent tagging by keyword rules (English, Hindi, Marathi), as in the prototype.
The DistilBERT tagger of Supplement S4 is evaluated separately (experiments/intent_tagger.py).
Keyword lists are starting points and need review by the counsellor panel."""
from __future__ import annotations
import re
from typing import Dict

INTENTS = ("academic_stress", "exam_anxiety", "social_conflict", "other")
RULES = {
    "exam_anxiety": ["exam", "test", "viva", "परीक्षा", "इम्तिहान", "परीक्षेची", "भीती", "nervous", "panic"],
    "academic_stress": ["assignment", "deadline", "backlog", "project", "submission", "attendance", "grades", "marks", "study",
                        "पढ़ाई", "अभ्यास", "प्रोजेक्ट", "अंतिम तारीख", "गुण"],
    "social_conflict": ["friend", "roommate", "argument", "fight", "bully", "ignored", "alone", "lonely",
                        "दोस्त", "झगड़ा", "अकेला", "मित्र", "भांडण", "एकटा"],
}


def tag_intent(text: str) -> Dict[str, float]:
    """Return one-hot-like probabilities over INTENTS (rule hits normalised)."""
    t = text.lower(); hits = {k: sum(1 for w in ws if re.search(re.escape(w), t)) for k, ws in RULES.items()}
    total = sum(hits.values())
    if total == 0:
        return {"academic_stress": 0.0, "exam_anxiety": 0.0, "social_conflict": 0.0, "other": 1.0}
    return {**{k: v / total for k, v in hits.items()}, "other": 0.0}
