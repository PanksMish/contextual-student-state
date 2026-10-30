"""Bounded session memory: the most recent k states and the interventions tried with their reactions.
Kept in RAM; persisted only if the student agrees (RuntimeConfig.persist_memory)."""
from __future__ import annotations
import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List
from ..state import StateRecord, MEMORY_K


class SessionMemory:
    def __init__(self, k: int = MEMORY_K):
        self.states: Deque[StateRecord] = deque(maxlen=k); self.interventions: List[Dict[str, str]] = []

    def add_state(self, rec: StateRecord) -> None: self.states.append(rec)

    def add_intervention(self, name: str, reaction: str = "unknown") -> None:
        self.interventions.append({"name": name, "reaction": reaction})

    def set_reaction(self, name: str, reaction: str) -> None:
        for it in reversed(self.interventions):
            if it["name"] == name:
                it["reaction"] = reaction; return

    def failed(self) -> set: return {i["name"] for i in self.interventions if i["reaction"] == "did_not_help"}

    def clear(self) -> None: self.states.clear(); self.interventions.clear()

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(dict(states=[s.__dict__ for s in self.states], interventions=self.interventions), default=list))
