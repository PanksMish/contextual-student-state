"""Stage timing harness for the cloud and offline paths (resource accounting section).

Provide callables for each stage; the harness records wall-clock time per stage over N runs and
reports medians and the total T = input + stress + context + LLM + output.

usage (as a library):
  from experiments.latency import measure
  res = measure({"input": f1, "stress": f2, "context": f3, "llm": f4, "output": f5}, runs=30)
"""
from __future__ import annotations
import time
import numpy as np


def measure(stages: dict, runs: int = 30, warmup: int = 2) -> dict:
    for _ in range(warmup):
        for fn in stages.values():
            fn()
    t = {k: [] for k in stages}
    for _ in range(runs):
        for k, fn in stages.items():
            s = time.perf_counter(); fn(); t[k].append(time.perf_counter() - s)
    med = {k: float(np.median(v)) for k, v in t.items()}
    totals = np.sum([t[k] for k in t], axis=0)
    return dict(median_by_stage=med, median_total=float(np.median(totals)), runs=runs)
