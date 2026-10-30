"""Multilingual DistilBERT intent tagger (Supplement S4).

Fine-tunes distilbert-base-multilingual-cased for four epochs at lr 5e-5 on the training split
and evaluates on the held-out, conversation-separated test split. Reports per-class precision,
recall, F1, macro F1, MCC, expected calibration error (15 bins) and macro F1 by language.

Input CSV: text, intent (academic_stress|exam_anxiety|social_conflict|other), language, split (train|test), conversation_id.

usage: python -m experiments.intent_tagger DATA.csv OUT.json [--seed 20260917]
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from .common import set_seed, macro_f1, save_json

CLASSES = ["academic_stress", "exam_anxiety", "social_conflict", "other"]


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def main(path, out, seed):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    set_seed(seed); df = pd.read_csv(path)
    assert not set(df[df.split == "train"].conversation_id) & set(df[df.split == "test"].conversation_id), "conversation leakage"
    name = "distilbert-base-multilingual-cased"
    tok = AutoTokenizer.from_pretrained(name); model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=4)
    y = df.intent.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    enc = lambda t: tok(list(t), padding=True, truncation=True, max_length=128, return_tensors="pt")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    idx = np.where(tr)[0]
    for _ in range(4):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 16):
            b = idx[i:i + 16]; e = enc(df.text.iloc[b])
            loss = model(**e, labels=torch.tensor(y[b])).loss; opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); probs = []
    with torch.no_grad():
        for i in range(0, te.sum(), 64):
            probs.append(torch.softmax(model(**enc(df.text[te].iloc[i:i + 64])).logits, -1).numpy())
    P = np.concatenate(probs); yt = y[te]; pred = P.argmax(1)
    p, r, f, s = precision_recall_fscore_support(yt, pred, labels=range(4), zero_division=0)
    res = dict(per_class={c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i])) for i, c in enumerate(CLASSES)},
               macro_f1=macro_f1(yt, pred), mcc=float(matthews_corrcoef(yt, pred)), ece=ece(P, yt),
               by_language={l: macro_f1(yt[(df.language[te] == l).to_numpy()], pred[(df.language[te] == l).to_numpy()]) for l in df.language.unique()})
    save_json(res, out); print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=20260917)
    a = ap.parse_args(); main(a.data, a.out, a.seed)
