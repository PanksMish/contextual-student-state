"""Language-model stage: cloud (OpenAI-compatible endpoint) or offline (4-bit Gemma-2B-it via llama.cpp)."""
from __future__ import annotations
import json, os, urllib.request
from typing import Protocol
from .config import CloudConfig, OfflineConfig
from ..state import CSS

SYSTEM = ("You are a supportive assistant for a university student. Reply in at most 150 words, in the student's language. "
          "Do not diagnose, do not give medical or medication advice, and if the student mentions a risk to their safety, "
          "encourage contact with the counselling cell or a trusted person.")

ACTION_HINT = {
    "grounding": "Offer one short grounding or breathing step before anything else.",
    "study_planning": "Help the student break the work into one or two small steps for today.",
    "problem_solving": "Help the student think through one practical step for the situation.",
    "acknowledge_emotion": "Acknowledge the feeling without assuming more than the student said.",
    "supportive_checkin": "Check in warmly and ask one open question.",
}


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


def build_prompt(css: CSS, action: str, utterance: str) -> str:
    """Structured fields as in condition C2, rendered from the live state."""
    from ..state import trend
    tr = trend(list(css.history), css.current) or "unknown"
    miss = css.missing()
    interv = "; ".join(f"{i['name']} ({i['reaction']})" for i in css.interventions) or "none"
    state = (f"stress {css.current.s:.2f} (confidence {css.current.c:.2f}{', tentative' if css.tentative() else ''}); trend {tr}"
             + (f"; unavailable: {', '.join(miss)}" if miss else ""))
    return (f"{SYSTEM}\n\n[State] {state}\n[Academic context] {css.academic_context or 'not provided'}\n"
            f"[Earlier interventions] {interv}\n[Plan] {ACTION_HINT.get(action, ACTION_HINT['supportive_checkin'])}\n"
            f"Student: {utterance}")


class CloudGenerator:
    def __init__(self, cfg: CloudConfig): self.cfg = cfg

    def generate(self, prompt: str) -> str:
        body = json.dumps(dict(model=self.cfg.model, temperature=0.7, top_p=0.9, max_tokens=260,
                               messages=[{"role": "user", "content": prompt}])).encode()
        req = urllib.request.Request(self.cfg.url, body, {"Content-Type": "application/json",
                                     "Authorization": "Bearer " + os.environ.get(self.cfg.api_key_env, "")})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()


class OfflineGenerator:
    def __init__(self, cfg: OfflineConfig):
        from llama_cpp import Llama
        self.llm = Llama(model_path=cfg.model_path, n_ctx=cfg.n_ctx, n_threads=cfg.n_threads, verbose=False)

    def generate(self, prompt: str) -> str:
        out = self.llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0.7, top_p=0.9, max_tokens=260)
        return out["choices"][0]["message"]["content"].strip()


class FallbackGenerator:
    """Tries the cloud path, falls back to the offline path on any failure (cloud-failure robustness test)."""
    def __init__(self, primary: Generator, fallback: Generator): self.p, self.f = primary, fallback

    def generate(self, prompt: str) -> str:
        try:
            return self.p.generate(prompt)
        except Exception:
            return self.f.generate(prompt)


def truncate_words(text: str, n: int = 150) -> str: return " ".join(text.split()[:n])
