"""Six-stage pipeline: input -> encoding -> stress estimate -> context state -> reply selection -> output.
Components are injected, so each can be replaced by a simulator in tests and demos.
Stage timings follow T_total = T_input + T_stress + T_context + T_LLM + T_output."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from ..state import CSS, StateRecord, expression_index
from .config import RuntimeConfig
from .generator import Generator, build_prompt, truncate_words
from .intent import tag_intent
from .memory import SessionMemory
from .safety import CounsellorQueue, detect_risk, guard_reply, risk_template
from .selector import select_action, ESCALATE


@dataclass
class Reply:
    text: str
    action: str
    reason: str
    css: CSS
    flags: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


class StudentSupportPipeline:
    def __init__(self, cfg: RuntimeConfig, fusion, generator: Generator, queue: Optional[CounsellorQueue] = None,
                 face=None, speech=None, session_id: str = "anon"):
        self.cfg, self.fusion, self.gen, self.queue = cfg, fusion, generator, queue or CounsellorQueue()
        self.face, self.speech, self.session = face, speech, session_id
        self.memory = SessionMemory(); self.t0 = time.time()

    def respond(self, utterance: str, *, window=None, frame=None, audio=None, academic_context: str = "",
                ctx_vec: Optional[np.ndarray] = None, language: Optional[str] = None) -> Reply:
        T = {}; tic = time.perf_counter()
        quality: Dict[str, float] = {}; feats: Dict[str, Optional[np.ndarray]] = {"wear": None, "face": None, "text": None}
        probs_face = None
        if audio is not None and self.speech is not None:
            utterance, q = self.speech.transcribe(audio, language); quality["text"] = q
        T["input"] = time.perf_counter() - tic; tic = time.perf_counter()
        if window is not None and window.available:
            feats["wear"], quality["wear"] = window.features, window.quality
        if frame is not None and self.face is not None:
            probs_face, qf = self.face(frame)
            if probs_face is not None:
                from .vision import CLASSES
                feats["face"], quality["face"] = np.array([probs_face[c] for c in CLASSES]), qf
        intent = tag_intent(utterance)
        from .intent import INTENTS
        feats["text"], quality["text"] = np.array([intent[k] for k in INTENTS]), quality.get("text", 1.0)
        out = self.fusion(feats, quality, ctx_vec if ctx_vec is not None else np.zeros(self._ctx_size()))
        T["stress"] = time.perf_counter() - tic; tic = time.perf_counter()
        rec = StateRecord((time.time() - self.t0) / 60.0, out.s, out.c, tuple(out.available), tuple(out.quality))
        css = CSS(rec, expression_index(probs_face) if probs_face else None, academic_context, utterance,
                  list(self.memory.interventions), list(self.memory.states))
        self.memory.add_state(rec)
        dec = select_action(css, utterance, intent, self.cfg.thresholds, self.memory.failed())
        T["context"] = time.perf_counter() - tic; tic = time.perf_counter()
        flags: List[str] = []
        if dec.action == ESCALATE:
            text = risk_template(self.cfg.resources)
            if self.cfg.notify_counsellor_on_risk:
                self.queue.add(self.session, "explicit safety cue", f"s={out.s:.2f}, c={out.c:.2f}", text)
            flags.append("needs_human_review")
            T["llm"] = 0.0
        else:
            text = truncate_words(self.gen.generate(build_prompt(css, dec.action, utterance)), self.cfg.max_words)
            T["llm"] = time.perf_counter() - tic
        tic = time.perf_counter()
        text, g = guard_reply(text, self.cfg.resources) if dec.action != ESCALATE else (text, [])
        if g:
            flags.append("guard_blocked"); self.queue.add(self.session, "guard blocked reply", f"s={out.s:.2f}", text)
        if dec.action != ESCALATE:
            self.memory.add_intervention(dec.action)
        T["output"] = time.perf_counter() - tic; T["total"] = sum(T.values())
        return Reply(text, dec.action, dec.reason, css, flags, T)

    def _ctx_size(self) -> int:
        return getattr(self.fusion, "sizes", {}).get("ctx", 0)
