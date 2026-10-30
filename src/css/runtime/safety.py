"""Safety layer: explicit-risk detection, reply guard and the counsellor queue.

Risk keywords are a conservative starting list (English, Hindi, Marathi) and must be reviewed by
the counsellor panel. A hit sends a fixed template, never generated text, and flags the exchange
for a human. The guard blocks diagnostic and medication language in generated replies."""
from __future__ import annotations
import re, sqlite3, time
from typing import List, Tuple
from .config import Resources

RISK = ["kill myself", "end my life", "end it all", "suicide", "suicidal", "want to die", "hurt myself", "self harm", "self-harm",
        "no reason to live", "better off dead", "आत्महत्या", "मरना चाहता", "मरना चाहती", "जान दे", "खुद को नुकसान",
        "मरायचं", "जीव द्यायचा", "स्वतःला इजा"]
GUARD = [r"\byou (have|suffer from|are suffering from)\b.{0,40}\b(depress|anxiety disorder|bipolar|adhd|ptsd|disorder)",
         r"\b(diagnos(e|is|ed))\b", r"\b\d+\s?(mg|milligram|ml)\b", r"\b(take|try|use)\b.{0,30}\b(tablet|pill|medication|antidepressant|sedative)\b"]


def detect_risk(text: str) -> Tuple[bool, List[str]]:
    t = text.lower(); hits = [k for k in RISK if k in t]
    return bool(hits), hits


def risk_template(res: Resources) -> str:
    lines = " ".join(f"{h}." for h in res.helplines)
    return (f"I am really glad you told me. What you describe matters, and you deserve support from a person right now. "
            f"Please contact {res.counselling_cell} or someone you trust today. {lines} If you may be in danger, call your local emergency number.")


def guard_reply(reply: str, res: Resources) -> Tuple[str, List[str]]:
    flags = [p for p in GUARD if re.search(p, reply, re.I)]
    if flags:
        return ("I want to be careful here. I cannot judge medical questions, and a counsellor can help with that. "
                f"Would you like to talk about what is worrying you, or reach {res.counselling_cell}?"), flags
    return reply, []


class CounsellorQueue:
    """Local SQLite queue of flagged exchanges for counsellor review (pseudonymous ids only)."""

    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS flags(id INTEGER PRIMARY KEY, ts REAL, session TEXT, reason TEXT, "
                        "state TEXT, reply TEXT, status TEXT DEFAULT 'open', note TEXT DEFAULT '')")

    def add(self, session: str, reason: str, state: str, reply: str) -> int:
        cur = self.db.execute("INSERT INTO flags(ts,session,reason,state,reply) VALUES(?,?,?,?,?)", (time.time(), session, reason, state, reply))
        self.db.commit(); return cur.lastrowid

    def open_items(self): return self.db.execute("SELECT id,ts,session,reason,state,reply FROM flags WHERE status='open' ORDER BY id").fetchall()

    def resolve(self, flag_id: int, note: str = "") -> None:
        self.db.execute("UPDATE flags SET status='resolved', note=? WHERE id=?", (note, flag_id)); self.db.commit()
