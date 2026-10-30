"""Runtime configuration. Every threshold here is a prototype choice (see paper Sections III-IV)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class Thresholds:
    elevated_s: float = 0.60       # stress estimate s_t in [0, 1]; 0-1 scale, never 0-100
    rising_eps: float = 0.10       # same value as css.state.TREND_EPS
    tau: float = 0.30              # confidence below this marks the state tentative
    expression_alert: float = 70.0 # expression index E_t in [0, 100] that triggers an empathic acknowledgement


@dataclass
class Resources:
    """Verify every entry with the institution before deployment."""
    counselling_cell: str = "the K. J. Somaiya counselling cell"
    helplines: List[str] = field(default_factory=lambda: ["Tele-MANAS 14416"])


@dataclass
class CloudConfig:
    url: str = "https://api.example.com/v1/chat/completions"   # any OpenAI-compatible endpoint
    model: str = "<hosted-model-name>"
    api_key_env: str = "CSS_API_KEY"
    timeout_s: float = 20.0


@dataclass
class OfflineConfig:
    model_path: str = "models/gemma-2b-it.Q4_K_M.gguf"       # 4-bit quantised Gemma-2B-it
    n_ctx: int = 2048
    n_threads: int = 4


@dataclass
class RuntimeConfig:
    thresholds: Thresholds = field(default_factory=Thresholds)
    resources: Resources = field(default_factory=Resources)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    max_words: int = 150
    window_s: float = 10.0
    step_s: float = 5.0
    use_cloud: bool = True
    notify_counsellor_on_risk: bool = True   # only with the student's consent (see paper Section IX)
    persist_memory: bool = False             # state is kept in RAM unless the student agrees
