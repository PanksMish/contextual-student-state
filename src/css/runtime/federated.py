"""Federated client and server for the linear head (paper Section IV, federated design).
Clients train locally on their own windows and send only the 36 head parameters (144 bytes, float32);
the server averages updates weighted by sample counts (FedAvg). FedAvg alone gives no privacy
guarantee (the paper states this); differential privacy and secure aggregation are future work."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


class Client:
    def __init__(self, X: np.ndarray, y: np.ndarray, n_classes: int = 4):
        self.X, self.y, self.k = X, y, n_classes

    def train(self, global_params: np.ndarray, lr=0.05, momentum=0.9, batch=256) -> Tuple[np.ndarray, int]:
        d = self.X.shape[1]; v = global_params.copy(); buf = np.zeros_like(v)
        for i in range(0, len(self.X), batch):
            W, b = v[: d * self.k].reshape(d, self.k), v[d * self.k:]
            p = _softmax(self.X[i:i + batch] @ W + b); p[np.arange(len(p)), self.y[i:i + batch]] -= 1
            g = np.concatenate([(self.X[i:i + batch].T @ p / len(p)).ravel(), p.mean(0)])
            buf = momentum * buf + g; v = v - lr * buf
        return v, len(self.X)


def fedavg(updates: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    return np.average([u for u, _ in updates], axis=0, weights=[n for _, n in updates])


def update_bytes(params: np.ndarray) -> int: return int(params.astype(np.float32).nbytes)
