# Code: Contextual Student State

Code for the paper **Context-Aware Multimodal Student Support with Wearable Sensing and Conversational AI** (Mishra, Parab, Saha, Hotelwala; K. J. Somaiya School of Engineering).

Analysis, experiments, prototype runtime and wearable firmware. Data tables, the pre-registration and the dataset specification are in the separate data repository **github.com/PanksMish/student-support-css** . This repository contains no data, raw facial images, audio, video, transcripts or student identifiers.

## Layout

| Path | Content |
|---|---|
| `src/css/state.py` | Contextual Student State: stress estimate, confidence c_t, expression index, quality scores, missing-modality handling, trend (paper Section III) |
| `src/css/conditions.py` | Prompt builders for conditions C0, C1, CF, C2 and R0, R1, R2 (Section VI) |
| `src/css/stats.py` | Scenario-level paired bootstrap, Holm correction, Wilcoxon test, non-inferiority bound, ordinal Krippendorff alpha |
| `src/css/runtime/`, `firmware/wearable/` | Prototype implementation: wearable firmware, sensing, fusion, selector, generator, safety layer, counsellor queue, federated client/server, Grad-CAM. See `docs/ARCHITECTURE.md`; demo: `python scripts/run_demo.py` |
| `experiments/` | Scripts for every component study (faces, WESAD, federated, ESConv proxy, fusion, intent, ASR, wearable validation, reply generation). Written from the paper's protocol and tested on synthetic data; see `experiments/README.md` |
| `scripts/analyze_ratings.py` | Rebuilds the human-rated tables (Table V, sensor-replay table, alpha, flags) from the CSV tables |
| `scripts/check_dataset.py` | Checks counts and freeze-before-generation against the design (240 / 180 / 120 responses, 3 raters) |
| `scripts/split_hash_audit.py` | SHA-256 and decoded-pixel audit of an image archive for cross-split duplicates and conflicting labels |
| `scripts/make_synthetic_demo.py` | Random data of the right shape for smoke tests. Never a result. |

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python scripts/make_synthetic_demo.py data/synthetic_demo
python scripts/check_dataset.py data/synthetic_demo
python scripts/analyze_ratings.py data/synthetic_demo results/synthetic_demo
```

Clone the data repository next to this one and point the scripts at its `data/released/` folder:

```bash
python scripts/check_dataset.py ../student-support-css/data/released
python scripts/analyze_ratings.py ../student-support-css/data/released ../student-support-css/results
```

## Status

The experiment scripts and the runtime were written from the paper's protocol and tested on synthetic fixtures only (see `experiments/README.md`, `docs/ARCHITECTURE.md`).

## Licence

MIT (`LICENSE`).
