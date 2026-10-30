#!/usr/bin/env python3
"""Audit an image archive for exact and decoded-pixel duplicates across splits.

usage: python scripts/split_hash_audit.py ARCHIVE_DIR OUT_CSV
ARCHIVE_DIR/{train,val,test}/<label>/*.png|jpg. Writes one row per file with SHA-256 of bytes,
SHA-256 of decoded pixels, split and label, then prints cross-split duplicate counts and
hashes that carry conflicting labels.
"""
import csv, hashlib, sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

root, out = Path(sys.argv[1]), sys.argv[2]
rows = []
for split in ("train", "val", "test"):
    for p in sorted((root / split).rglob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"): continue
        b = p.read_bytes()
        px = Image.open(p).convert("RGB")
        rows.append(dict(path=str(p.relative_to(root)), split=split, label=p.parent.name,
                         sha_bytes=hashlib.sha256(b).hexdigest(),
                         sha_pixels=hashlib.sha256(px.tobytes() + str(px.size).encode()).hexdigest()))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for key in ("sha_bytes", "sha_pixels"):
    g = defaultdict(list)
    for r in rows: g[r[key]].append(r)
    cross = sum(1 for v in g.values() if len({r["split"] for r in v}) > 1)
    conflict = sum(1 for v in g.values() if len({r["label"] for r in v}) > 1)
    print(f"{key}: files {len(rows)}, unique {len(g)}, cross-split groups {cross}, conflicting-label groups {conflict}")
