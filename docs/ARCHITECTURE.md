# Prototype implementation

Maps the six stages of the paper's architecture figure to code. Components are injected into the pipeline, so each can be replaced by a simulator in tests.

| Stage | Module | Notes |
|---|---|---|
| 1. Input | `firmware/wearable/wearable.ino`, `runtime/wearable_io.py`, `runtime/vision.py`, `runtime/audio.py` | XIAO ESP32S3 with MAX30102, Grove GSR, MAX30205, MPU6050 sends one JSON line per second; 10-s windows, 5-s step; face frames and audio are processed live and not stored |
| 2. Encoding | `vision.FaceExpressionEncoder`, `audio.SpeechEncoder`, `intent.tag_intent` | Frozen Swin-T (or ResNet) with a linear head gives 7 expression probabilities; Whisper-small transcribes; keyword rules tag intent at runtime |
| 3. Stress estimate | `runtime/fusion.py`, `css/state.py` | MLP over features and availability mask; s_t = 0.5 p_mod + p_high; c_t = (1 - H/log K) x mean(a q) |
| 4. Context state | `css/state.py`, `runtime/memory.py` | CSS with change, history (k = 10), expression index, interventions with reactions; missing modalities recorded, never imputed |
| 5. Reply selection | `runtime/selector.py`, `runtime/generator.py` | Priority rules, then the language model (cloud, offline Gemma-2B-it, or cloud with offline fallback) |
| 6. Output and oversight | `runtime/safety.py`, `runtime/pipeline.py` | Fixed template and counsellor flag on explicit risk; guard against diagnostic and medication language; SQLite queue for counsellor review |
| Federated design | `runtime/federated.py` | Linear head only, 144-byte updates, FedAvg |
| Explainability | `runtime/xai.py` | Grad-CAM overlay and deletion curve |

## Hardware wiring (XIAO ESP32S3)

| Device | Pins |
|---|---|
| MAX30102, MAX30205, MPU6050 (I2C) | SDA D4 (GPIO5), SCL D5 (GPIO6), 3V3, GND |
| Grove GSR | signal A0 (D0), 3V3, GND |

## Training the models the runtime loads

1. Face head: `python -m experiments.fer_linear_head ...` (save the chosen head's `state_dict`; the script reports metrics, add a `torch.save` for deployment).
2. Fusion: `python -m experiments.fusion_cv windows.csv out.json --save-model fusion.pt`.
3. Offline language model: download a 4-bit Gemma-2B-it GGUF and set `OfflineConfig.model_path`.

## Status and limits

- Written from the paper's description. The simulator-based tests pass; the firmware, camera, microphone, Whisper, llama.cpp and cloud paths were not run on hardware or with real models.
- Keyword lists for risk and intent, thresholds in `runtime/config.py`, helpline numbers and the counselling-cell contact are starting points. The counsellor panel must review them, and every helpline must be verified before any use with students.
- The selector, offline stage and keyword tagger have no reported quality metrics in the paper.
- The system is a research prototype, not a clinical tool, and must not be deployed to students without ethics clearance, consent procedures and counsellor oversight.
