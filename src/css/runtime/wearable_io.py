"""Wearable stream: JSON lines from the ESP32S3 -> 10-s windows with quality scores.

Line format (1 Hz or faster):
  {"t": ms, "hr": bpm, "spo2": %, "gsr": uS, "temp": C, "ax": g, "ay": g, "az": g, "ir": raw}
hr == 0 or ir below IR_CONTACT means no finger contact; motion above MOTION_G flags the sample.
"""
from __future__ import annotations
import json, math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterator, List, Optional
import numpy as np

IR_CONTACT = 50_000
MOTION_G = 0.15
FEATURES = ("hr_mean", "hr_std", "spo2_mean", "gsr_mean", "gsr_slope", "temp_mean", "motion", "flagged")


@dataclass
class Window:
    t_start: float
    features: np.ndarray          # len(FEATURES)
    quality: float                # 1 - share of flagged samples
    available: bool


def parse_line(line: str) -> Optional[Dict[str, float]]:
    try:
        d = json.loads(line)
        return {k: float(d[k]) for k in ("t", "hr", "spo2", "gsr", "temp", "ax", "ay", "az", "ir")}
    except (ValueError, KeyError, TypeError):
        return None


def sample_flagged(s: Dict[str, float], prev_mag: Optional[float]) -> bool:
    mag = math.sqrt(s["ax"] ** 2 + s["ay"] ** 2 + s["az"] ** 2)
    moving = prev_mag is not None and abs(mag - prev_mag) > MOTION_G
    return s["hr"] <= 0 or s["ir"] < IR_CONTACT or moving


def summarise(samples: List[Dict[str, float]]) -> Window:
    flags, prev = [], None
    mags = []
    for s in samples:
        flags.append(sample_flagged(s, prev)); prev = math.sqrt(s["ax"] ** 2 + s["ay"] ** 2 + s["az"] ** 2); mags.append(prev)
    flags = np.array(flags); good = [s for s, f in zip(samples, flags) if not f]
    quality = float(1.0 - flags.mean())
    if len(good) < max(3, len(samples) // 3):
        return Window(samples[0]["t"] / 1000.0, np.zeros(len(FEATURES)), quality, False)
    hr = np.array([s["hr"] for s in good]); g = np.array([s["gsr"] for s in good]); t = np.array([s["t"] for s in good]) / 1000.0
    slope = float(np.polyfit(t - t[0], g, 1)[0]) if len(good) > 2 and t[-1] > t[0] else 0.0
    f = np.array([hr.mean(), hr.std(), np.mean([s["spo2"] for s in good]), g.mean(), slope,
                  np.mean([s["temp"] for s in good]), float(np.std(mags)), float(flags.mean())])
    return Window(samples[0]["t"] / 1000.0, f, quality, True)


class WindowBuffer:
    def __init__(self, window_s: float = 10.0, step_s: float = 5.0):
        self.window_s, self.step_s = window_s, step_s
        self.buf: Deque[Dict[str, float]] = deque(); self.next_emit: Optional[float] = None

    def push(self, s: Dict[str, float]) -> Optional[Window]:
        self.buf.append(s); t = s["t"] / 1000.0
        while self.buf and t - self.buf[0]["t"] / 1000.0 > self.window_s:
            self.buf.popleft()
        if self.next_emit is None:
            self.next_emit = self.buf[0]["t"] / 1000.0 + self.window_s
        if t >= self.next_emit and len(self.buf) > 2:
            self.next_emit += self.step_s
            return summarise(list(self.buf))
        return None


def read_serial(port: str, baud: int = 115200) -> Iterator[Dict[str, float]]:
    import serial  # pyserial
    with serial.Serial(port, baud, timeout=2) as ser:
        while True:
            s = parse_line(ser.readline().decode("utf-8", "ignore"))
            if s:
                yield s


def simulate(seconds: int = 60, hr: float = 78.0, seed: int = 0, contact: bool = True) -> Iterator[Dict[str, float]]:
    """Synthetic stream for demos and tests. Not data; never used in any reported result."""
    rng = np.random.default_rng(seed)
    for i in range(seconds):
        yield dict(t=i * 1000.0, hr=(hr + rng.normal(0, 1.5)) if contact else 0.0, spo2=97 + rng.normal(0, 0.3),
                   gsr=2.0 + 0.01 * i + rng.normal(0, 0.02), temp=33.5 + rng.normal(0, 0.05),
                   ax=rng.normal(0, 0.01), ay=rng.normal(0, 0.01), az=1.0 + rng.normal(0, 0.01), ir=80_000 if contact else 0)
