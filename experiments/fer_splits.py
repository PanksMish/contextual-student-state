"""Pixel-disjoint split of the project facial archive (Section V-C) and the official FER-2013 loader.

usage:
  python -m experiments.fer_splits pixel-disjoint AUDIT_CSV OUT_CSV [--seed 20260930]
  python -m experiments.fer_splits official FER2013_CSV OUT_NPZ

AUDIT_CSV comes from scripts/split_hash_audit.py (columns path, split, label, sha_bytes, sha_pixels).

Rules, as in the paper:
 1. Group files by decoded-pixel hash; drop groups whose copies carry conflicting labels.
 2. Collapse each remaining group to one image.
 3. Groups that occur in the original test partition form the test set; every such group is
    removed from the training pool, even if copies also sit in train or validation.
 4. The remaining groups are split 80/20, stratified by label, into train and validation.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def pixel_disjoint(audit: pd.DataFrame, seed: int = 20260930) -> pd.DataFrame:
    g = audit.groupby("sha_pixels")
    conflict = g["label"].nunique() > 1
    keep = audit[~audit.sha_pixels.map(conflict)]
    in_test = set(keep.loc[keep.split == "test", "sha_pixels"])
    first = keep.sort_values("path").groupby("sha_pixels", as_index=False).first()
    first["assigned"] = np.where(first.sha_pixels.isin(in_test), "test", "pool")
    pool = first[first.assigned == "pool"]
    tr, va = train_test_split(pool, test_size=0.2, stratify=pool.label, random_state=seed)
    first.loc[tr.index, "assigned"] = "train"
    first.loc[va.index, "assigned"] = "val"
    return first[["path", "label", "sha_pixels", "assigned"]]


def official_fer2013(csv_path: str):
    """Kaggle fer2013.csv (emotion, pixels, Usage) -> arrays and the three official partitions."""
    df = pd.read_csv(csv_path)
    x = np.stack([np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in df.pixels])
    y = df.emotion.to_numpy()
    part = df.Usage.map({"Training": "train", "PublicTest": "public_test", "PrivateTest": "private_test"}).to_numpy()
    return x, y, part


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pixel-disjoint"); a.add_argument("audit"); a.add_argument("out"); a.add_argument("--seed", type=int, default=20260930)
    b = sub.add_parser("official"); b.add_argument("csv"); b.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "pixel-disjoint":
        res = pixel_disjoint(pd.read_csv(args.audit), args.seed)
        res.to_csv(args.out, index=False)
        print(res.assigned.value_counts().to_string())
    else:
        x, y, part = official_fer2013(args.csv)
        np.savez_compressed(args.out, x=x, y=y, part=part)
        print({p: int((part == p).sum()) for p in np.unique(part)})
