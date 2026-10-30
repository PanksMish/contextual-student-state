"""Word error rate of offline Whisper-small by language (Section V-B and Table on-device).

Normalisation: Unicode NFC, lower-case, punctuation removed; Hindi and Marathi are scored in
Devanagari without transliteration. Speaker-level bootstrap for intervals.

usage:
  python -m experiments.asr_wer transcribe MANIFEST.csv HYP.csv     # needs openai-whisper or transformers
  python -m experiments.asr_wer score HYP.csv OUT.json
MANIFEST.csv: audio_path, language (en|hi|mr), speaker, reference
HYP.csv adds hypothesis.
"""
from __future__ import annotations
import argparse, re, unicodedata
import numpy as np, pandas as pd
from .common import cluster_bootstrap, save_json


def normalise(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t)).lower()
    t = "".join(ch if (ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M")) else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalise(ref).split(), normalise(hyp).split()
    return edit_distance(r, h), len(r)


def transcribe(manifest: str, out: str, model_name: str = "small"):
    import whisper
    m = whisper.load_model(model_name, device="cpu")
    df = pd.read_csv(manifest)
    df["hypothesis"] = [m.transcribe(p, language=l, fp16=False)["text"] for p, l in zip(df.audio_path, df.language)]
    df.to_csv(out, index=False)


def score(hyp_csv: str, out: str):
    df = pd.read_csv(hyp_csv); res = {}
    for lang, g in df.groupby("language"):
        e = np.array([wer(r, h) for r, h in zip(g.reference, g.hypothesis)])
        clusters = {s: (e[(g.speaker == s).to_numpy()][:, 0], e[(g.speaker == s).to_numpy()][:, 1]) for s in g.speaker.unique()}
        ci = cluster_bootstrap(clusters, lambda er, n: er.sum() / n.sum(), 5000, 0)
        res[lang] = dict(wer=float(e[:, 0].sum() / e[:, 1].sum()), ci=ci, n_utts=len(g), n_speakers=g.speaker.nunique())
    save_json(res, out); print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe"); t.add_argument("manifest"); t.add_argument("out")
    s = sub.add_parser("score"); s.add_argument("hyp"); s.add_argument("out")
    a = ap.parse_args()
    transcribe(a.manifest, a.out) if a.cmd == "transcribe" else score(a.hyp, a.out)
