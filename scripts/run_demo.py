#!/usr/bin/env python3
"""Console demo of the full pipeline without hardware: simulated wearable, no camera, typed input.
Uses a stub fusion model and a canned generator unless real components are configured.
Nothing printed here is a result.

usage: python scripts/run_demo.py [--serial /dev/ttyACM0] [--cloud]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from css.runtime.config import RuntimeConfig
from css.runtime.fusion import StubFusion
from css.runtime.pipeline import StudentSupportPipeline
from css.runtime.wearable_io import WindowBuffer, simulate, read_serial


class CannedGenerator:
    def generate(self, prompt: str) -> str:
        return "That sounds heavy. Let us take one small step first: breathe in for four counts and out for six, twice. What feels most pressing for today?"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--serial"); ap.add_argument("--cloud", action="store_true")
    a = ap.parse_args(); cfg = RuntimeConfig()
    gen = CannedGenerator()
    if a.cloud:
        from css.runtime.generator import CloudGenerator; gen = CloudGenerator(cfg.cloud)
    pipe = StudentSupportPipeline(cfg, StubFusion(), gen)
    buf = WindowBuffer(cfg.window_s, cfg.step_s); win = None
    for s in (read_serial(a.serial) if a.serial else simulate(30)):
        w = buf.push(s); win = w or win
    print("Type a message (empty line to stop).")
    while True:
        u = input("student> ").strip()
        if not u: break
        r = pipe.respond(u, window=win, academic_context="demo context")
        print(f"[{r.action}: {r.reason}; flags={r.flags}; total {r.timings['total']*1000:.0f} ms]\n{r.text}\n")
    for item in pipe.queue.open_items(): print("COUNSELLOR QUEUE:", item)


if __name__ == "__main__":
    main()
