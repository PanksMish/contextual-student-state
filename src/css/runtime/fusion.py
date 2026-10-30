"""Fusion stage: MLP over [wearable summary, face probabilities, intent probabilities, context, mask]
-> class probabilities over the PSS-10 bands -> s_t and c_t (css.state). Train with
experiments.fusion_cv.train_final and save with the metadata below."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import numpy as np
from ..state import stress_estimate, confidence

MODS = ("wear", "face", "text")


@dataclass
class FusionOutput:
    probs: np.ndarray; s: float; c: float; available: Sequence[int]; quality: Sequence[float]


class FusionModel:
    def __init__(self, path: str):
        import torch
        ck = torch.load(path, map_location="cpu")
        self.sizes = ck["sizes"]            # dict: wear, face, text, ctx
        d = sum(self.sizes.values()) + 3
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(64, 3))
        self.net.load_state_dict(ck["state_dict"]); self.net.eval(); self.torch = torch

    def __call__(self, feats: Dict[str, Optional[np.ndarray]], quality: Dict[str, float], ctx: np.ndarray) -> FusionOutput:
        parts, avail, q = [], [], []
        for m in MODS:
            x = feats.get(m); a = 0 if x is None else 1
            parts.append(np.zeros(self.sizes[m]) if x is None else np.asarray(x, float)); avail.append(a); q.append(quality.get(m, 0.0) if a else 0.0)
        vec = np.concatenate(parts + [ctx, np.array(avail, float)])
        with self.torch.no_grad():
            p = self.torch.softmax(self.net(self.torch.tensor(vec, dtype=self.torch.float32)), -1).numpy()
        return FusionOutput(p, stress_estimate(*p), confidence(p, avail, q), avail, q)


class StubFusion:
    """Fixed output for tests and demos. Not an estimator."""
    def __init__(self, probs=(0.2, 0.5, 0.3)): self.p = np.array(probs)

    def __call__(self, feats, quality, ctx) -> FusionOutput:
        avail = [0 if feats.get(m) is None else 1 for m in MODS]; q = [quality.get(m, 0.0) if a else 0.0 for m, a in zip(MODS, avail)]
        return FusionOutput(self.p, stress_estimate(*self.p), confidence(self.p, avail, q), avail, q)
