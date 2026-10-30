"""Prompt construction for the rating conditions (Section VI).

C0  utterance only
C1  + detector output
CF  + the information of C2 as one unstructured paragraph (representation control)
C2  + structured context and history
R0/R1/R2 (sensor study) reuse the same fields with state from nothing / the
fusion model / the reference state.
"""
from __future__ import annotations
from typing import Dict

SYSTEM = (
    "You are a supportive assistant for a university student. Reply in at most "
    "150 words, in the student's language. Do not diagnose, do not give medical "
    "or medication advice, and if the student mentions a risk to their safety, "
    "encourage contact with the counselling cell or a trusted person."
)


def prompt(condition: str, sc: Dict[str, str]) -> str:
    u = sc["utterance"]
    if condition == "C0":
        return f"{SYSTEM}\n\nStudent: {u}"
    if condition == "C1":
        return f"{SYSTEM}\n\nDetector output: {sc['detector_output']}\nStudent: {u}"
    if condition == "CF":
        return f"{SYSTEM}\n\nDetector output: {sc['detector_output']}\nBackground: {sc['flat_text_version']}\nStudent: {u}"
    if condition == "C2":
        return (f"{SYSTEM}\n\nDetector output: {sc['detector_output']}\n"
                f"[Academic context] {sc['academic_context']}\n"
                f"[Stress trend and history] {sc['history_summary']}\n"
                f"[Earlier interventions] {sc['prior_intervention']} ({sc.get('prior_reaction') or 'n/a'})\n"
                f"Student: {u}")
    raise ValueError(condition)


def sensor_prompt(condition: str, sess: Dict[str, str]) -> str:
    base = f"{SYSTEM}\n\n[Academic context] {sess['academic_context']}\nStudent: {sess['transcript_text']}"
    if condition == "R0":
        return base
    band = sess["predicted_band"] if condition == "R1" else sess["reference_band"]
    s = f"\n[State] stress band: {band}"
    if condition == "R1":
        s += f"; confidence {float(sess['predicted_c']):.2f}"
        if sess.get("missing_modalities", "none") != "none":
            s += f"; unavailable: {sess['missing_modalities']}"
    return base + s
