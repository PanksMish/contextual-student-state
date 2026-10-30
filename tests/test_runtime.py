import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from css.state import CSS, StateRecord
from css.runtime.config import RuntimeConfig, Thresholds
from css.runtime.fusion import StubFusion
from css.runtime.pipeline import StudentSupportPipeline
from css.runtime.safety import detect_risk, guard_reply, risk_template, CounsellorQueue
from css.runtime.selector import select_action, ESCALATE, GROUND, PLAN, SOLVE, ACKNOWLEDGE, CHECKIN
from css.runtime.intent import tag_intent
from css.runtime.wearable_io import WindowBuffer, simulate, summarise
from css.runtime.federated import Client, fedavg, update_bytes
from css.runtime.memory import SessionMemory
from css.runtime.config import Resources

TH = Thresholds()
def mk(s=0.3, c=0.8, hist=(), e=None):
    return CSS(StateRecord(10, s, c), e, "", "", [], list(hist))

class Gen:
    def __init__(self, text="Let us take a breath together."): self.text, self.prompts = text, []
    def generate(self, p): self.prompts.append(p); return self.text

def test_risk_detection_multilingual():
    assert detect_risk("I want to die")[0] and detect_risk("मुझे आत्महत्या के विचार आ रहे हैं")[0] and detect_risk("मला जीव द्यायचा आहे")[0]
    assert not detect_risk("my exam is tomorrow")[0]

def test_selector_priority_order():
    i = tag_intent("my exam is tomorrow")
    assert select_action(mk(0.9), "I want to die", i, TH).action == ESCALATE          # safety first
    assert select_action(mk(0.8), "my exam is tomorrow", i, TH).action == GROUND       # elevated before planning
    assert select_action(mk(0.3), "my exam is tomorrow", i, TH).action == PLAN
    assert select_action(mk(0.3), "I fought with my roommate", tag_intent("I fought with my roommate"), TH).action == SOLVE
    assert select_action(mk(0.3, e=80), "hello", tag_intent("hello"), TH).action == ACKNOWLEDGE
    assert select_action(mk(0.3), "hello", tag_intent("hello"), TH).action == CHECKIN

def test_tentative_state_does_not_trigger_stress_rule_and_failed_actions_skipped():
    assert select_action(mk(0.9, c=0.1), "hello", tag_intent("hello"), TH).action == CHECKIN
    assert select_action(mk(0.9), "hello", tag_intent("hello"), TH, {GROUND}).action == CHECKIN

def test_rising_trend_triggers_grounding():
    h = [StateRecord(0, 0.2, 0.8)]
    assert select_action(mk(0.45, hist=h), "hello", tag_intent("hello"), TH).action == GROUND

def test_guard_blocks_diagnosis_and_dosage():
    r = Resources()
    assert guard_reply("You have an anxiety disorder.", r)[1] and guard_reply("Take 20 mg of it.", r)[1]
    assert not guard_reply("Let us breathe slowly.", r)[1]
    assert "Tele-MANAS" in risk_template(r)

def test_pipeline_normal_and_escalation_paths():
    gen = Gen(); pipe = StudentSupportPipeline(RuntimeConfig(), StubFusion(), gen)
    buf = WindowBuffer(); win = None
    for s in simulate(30):
        w = buf.push(s); win = w or win
    r = pipe.respond("my exam is tomorrow and I am panicking", window=win, academic_context="3rd year")
    assert r.action in (GROUND, PLAN) and gen.prompts and "[State]" in gen.prompts[0] and len(r.text.split()) <= 150
    assert set(r.timings) >= {"input", "stress", "context", "llm", "output", "total"}
    n = len(gen.prompts); r2 = pipe.respond("I want to end my life")
    assert r2.action == ESCALATE and len(gen.prompts) == n and "needs_human_review" in r2.flags and pipe.queue.open_items()
    pipe.queue.resolve(pipe.queue.open_items()[0][0], "called student"); assert not pipe.queue.open_items()

def test_missing_modalities_recorded_not_imputed():
    gen = Gen(); pipe = StudentSupportPipeline(RuntimeConfig(), StubFusion(), gen)
    r = pipe.respond("hello", window=None)
    assert r.css.missing() == ["wearable", "face"] and "unavailable: wearable, face" in gen.prompts[0]

def test_guarded_generation_is_replaced_and_flagged():
    pipe = StudentSupportPipeline(RuntimeConfig(), StubFusion(), Gen("You have depression. Take 50 mg daily."))
    r = pipe.respond("hello"); assert "guard_blocked" in r.flags and "mg" not in r.text

def test_wearable_windows_and_contact_loss():
    buf = WindowBuffer(10, 5); ws = [w for s in simulate(40) if (w := buf.push(s))]
    assert ws and all(w.available and w.quality > 0.8 for w in ws)
    nb = WindowBuffer(10, 5); wn = [w for s in simulate(40, contact=False) if (w := nb.push(s))]
    assert wn and not any(w.available for w in wn)

def test_federated_update_size_and_fedavg():
    rng = np.random.default_rng(0); g = np.zeros(36)
    c = [Client(rng.normal(size=(200, 8)), rng.integers(0, 4, 200)) for _ in range(3)]
    ups = [x.train(g) for x in c]; new = fedavg(ups)
    assert update_bytes(new) == 144 and new.shape == (36,) and not np.allclose(new, g)

def test_memory_bounded_and_failed_tracking():
    m = SessionMemory(k=3)
    for i in range(5): m.add_state(StateRecord(i, .1, .9))
    assert len(m.states) == 3
    m.add_intervention("grounding"); m.set_reaction("grounding", "did_not_help"); assert m.failed() == {"grounding"}
