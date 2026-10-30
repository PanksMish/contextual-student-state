"""Voice branch: record, estimate SNR, transcribe offline with Whisper-small."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from ..state import quality_audio


def estimate_snr_db(x: np.ndarray, sr: int = 16000, frame_s: float = 0.03) -> float:
    n = int(sr * frame_s); frames = x[: len(x) // n * n].reshape(-1, n)
    e = (frames ** 2).mean(1) + 1e-12; e.sort()
    k = max(1, len(e) // 10)
    return float(10 * np.log10(e[-k:].mean() / e[:k].mean()))


class SpeechEncoder:
    def __init__(self, model: str = "small", device: str = "cpu"):
        import whisper
        self.m = whisper.load_model(model, device=device)

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Tuple[str, float]:
        out = self.m.transcribe(audio.astype(np.float32), language=language, fp16=False)
        return out["text"].strip(), quality_audio(estimate_snr_db(audio))


def record(seconds: float = 10.0, sr: int = 16000) -> np.ndarray:
    import sounddevice as sd
    x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32"); sd.wait()
    return x[:, 0]
