# Experiments

Scripts for every component study in the paper. They were written from the protocol described in the paper and are tested on synthetic fixtures (`tests/test_experiments.py`). They have **not** been run on the real data. Run each on the real inputs, read the output, and replace the paper's estimated values only with what the run produces.

| Paper section | Script | Input | Seeds | Paper target (replace with your run) |
|---|---|---|---|---|
| V-C pixel-disjoint faces | `scripts/split_hash_audit.py` then `fer_splits pixel-disjoint`, then `fer_linear_head` | archive folders | split 20260930; heads 9301-9303 | Swin-T 0.6240 vs ResNet50 0.5579 acc |
| V-D official FER-2013 | `fer_splits official`, `fer_linear_head --test-part private_test` | `fer2013.csv` (Kaggle) | 9701-9703 | ResNet50 0.468, Swin-T 0.531 |
| V-E federated | `federated_benchmark` | window `.npz` from the HR/HRV mirror | 20260921-25 | central 0.5481, FedAvg 0.5519 |
| V-F WESAD raw | `wesad_raw features`, `wesad_raw ablate` | WESAD `S*/S*.pkl` | 20260915 | temporal 0.649 vs 0.618 (13/15 subjects) |
| V-G likelihood proxy | `esconv_likelihood` | `ESConv.json` | 20260920 | L1-L0 = -0.3666 nats/token |
| V-B fusion | `fusion_cv` | `windows.csv` (65 students) | 20260918 | full model 0.652 macro F1 |
| S4 intent | `intent_tagger` | intent CSV | 20260917 | macro F1 0.86 |
| S4 wearable | `wearable_validation` | pairs CSV | - | HR MAE 1.6 bpm |
| WER | `asr_wer transcribe`, `asr_wer score` | audio manifest | - | 0.08 / 0.14 / 0.21 |
| Latency | `latency.measure` | stage callables | - | 3.8 s cloud, 13.5 s offline |
| VI replies | `generate_replies`, then `scripts/analyze_ratings.py` | scenario and session tables | 20261008 | see paper |

## Order of use

1. `pip install -r requirements.txt -r requirements-ml.txt`
2. Build splits from the archive and the official FER-2013 CSV; run the heads.
3. Build the WESAD features and run the ablation; build the mirror windows and run the federated benchmark.
4. Run the ESConv proxy, intent tagger, ASR and wearable validation on the collected tables.
5. Freeze the scenarios and deposit the pre-registration package. Only then run `generate_replies`.
6. After rating, run `scripts/check_dataset.py` and `scripts/analyze_ratings.py`.

## Prototype choices to confirm against the paper text

- WESAD windows: 60 s, step 30 s, label purity 0.9; spectral bands VLF 0.003-0.04, LF 0.04-0.15, HF 0.15-0.4 Hz from a 4 Hz resampled RR series.
- BiLSTM: hidden 24 per direction, LayerNorm(48), Linear(48, 4) gives exactly 6,820 parameters (asserted in `tests/test_torch_models.py`).
- Temporal baseline: the same network on the current window only.
- Federated step: one pass over the pooled training windows (central) or one round of one local pass per client (FedAvg). Update traffic counts one upload and one download per client and round (12,960 bytes).
- Fusion: one hidden layer of 64 units, dropout 0.3, modality dropout 0.3, 40 epochs.
If the project's own scripts used different values, change the constants here and in the paper together.
