# Cutlery Classifier MVP

Offline inference MVP for classifying cutlery (`fork`, `knife`, `spoon`), written as an edge deployment case study. The point of this repo is to be honest about what’s here today and what’s still missing before it can run on a Raspberry Pi with a camera.

## Overview
- What exists: file-based inference + offline evaluation scripts (Python package + scripts).
- What doesn’t: camera loop, ONNXRuntime runner, actuator integration.
- Target (roadmap): Raspberry Pi (CPU) with ONNX Runtime.
- Constraints: end-to-end latency/jitter, preprocessing parity, reject handling, and reliable logging/safe defaults.

Intended runtime: camera → preprocessing → model → decision → actuation.

## What exists today

- Inference: `CutleryInferencer` in `src/cutlery_classifier/inference/inferencer.py`
  - Loads a checkpoint with `config`, `class_names`, `model_state_dict`.
  - Runs preprocessing + model forward pass.
  - Returns top-k softmax probabilities and `inference_time_ms` (see Performance).
- Models: `src/cutlery_classifier/models/factory.py`
  - `resnet18` and `mobilenet_v2`
  - adapts pretrained RGB weights to grayscale.
- Evaluation:
  - `scripts/evaluate_on_test_set.py` writes `results/evaluation/metrics.json` and `results/evaluation/confusion_matrix.png`
  - `scripts/test_dataset_inference.py` is an alternative evaluator that writes `outputs/run_<timestamp>/` artifacts and can do basic confidence/stress analysis (offline only).
- Export: `scripts/export_model.py` exports to ONNX/TorchScript (export only; no ONNXRuntime runner in this repo yet).

## What’s missing for a Pi deployment

To turn this into a real edge system:
- a camera capture loop (buffering + dropped-frame policy)
- an ONNXRuntime runner on ARM CPU, with preprocessing matched to the PyTorch path
- reject handling wired into runtime (confidence/margin, plus a clear “reject lane” behavior)
- an actuator interface (GPIO/serial/PLC) with a safe default state
- end-to-end latency/jitter measurement (capture→actuate, under load and at thermal steady state)

## Practical details

### Preprocessing (as implemented)

Base transforms are defined in `src/cutlery_classifier/data/transforms.py` and used for offline inference:
- resize to `image_size` (default `[320, 320]` from config)
- grayscale (`num_output_channels=1`)
- `ToTensor`
- normalize with mean `0.449`, std `0.226`

Notes:
- The older README mentioned a center crop to 224×224. That is not in the base transforms.
- Some scripts still define their own transforms. Before deploying, this needs to collapse to one canonical preprocessing path (PyTorch + ONNXRuntime).

### Decision / reject handling (planned)

In production, `argmax` is not enough. This needs explicit reject handling (low confidence, low margin, and/or simple multi-frame confirmation).

Right now, inference returns probabilities but does not apply a reject policy in the runtime path.

## Performance

What’s measured today:
- `CutleryInferencer.predict()` reports `inference_time_ms` measured around preprocessing + forward pass + postprocessing in Python.

What’s still missing (edge-relevant):
- Raspberry Pi CPU numbers
- ONNXRuntime latency on ARM
- end-to-end (capture→actuate) latency and jitter

Pi placeholders (to be replaced with real measurements):
- model-only ONNXRuntime latency (ms): TBD
- preprocess latency (ms): TBD
- end-to-end latency (ms): TBD
- sustained FPS at thermal steady state: TBD

## Reproducible run

Setup:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

pip install -e .
```

Offline single-image inference (PyTorch checkpoint):

```bash
cutlery-inference --device cpu --image <path/to/image.jpg> --model <path/to/checkpoint.pth>
```

Notes:
- The checkpoint must include `config`, `class_names`, and `model_state_dict` (see `CutleryInferencer.load_model()`).
- If you don’t have the checkpoint and dataset folders locally, these commands will fail. This repo does not guarantee those artifacts are present.

Offline evaluation on a directory-structured test set:

```bash
python scripts/evaluate_on_test_set.py \
  --device cpu \
  --model <path/to/checkpoint.pth> \
  --test-dir <path/to/test_dir>
```

Outputs:
- `results/evaluation/metrics.json`
- `results/evaluation/confusion_matrix.png`

## Gaps and next steps

Known gaps / risks (short list):
- missing camera loop
- missing actuator integration
- reject handling not wired into runtime
- preprocessing parity risk (multiple transform definitions in scripts)
- dependency bloat (dev tooling mixed into core requirements)

Next steps checklist:
- [ ] add a minimal runtime skeleton (camera loop + runner + actuator interface)
- [ ] add ONNXRuntime inference on ARM and lock preprocessing parity vs PyTorch
- [ ] implement reject handling and log why each decision was accepted/rejected
- [ ] build a Pi end-to-end latency/jitter harness (with thermal steady-state runs)
- [ ] split runtime vs dev dependencies so a Pi install is small and predictable

## License

MIT License (see `LICENSE`).
## Cutlery Classifier MVP — README v3 (human engineer pass)

This repo is an **offline inference MVP** for classifying cutlery into `fork`, `knife`, `spoon`. It’s written as a practical “edge deployment case study”: what works today, what’s missing, and what I’d build next to run it on a Raspberry Pi + camera.

### Overview
- **What exists**: file-based inference + offline evaluation scripts, packaged as a Python module.
- **What doesn’t**: no camera capture loop, no ONNXRuntime runner, no actuator integration.
- **Target (roadmap)**: Raspberry Pi (CPU) + ONNX Runtime baseline.
- **Constraints I care about**: end-to-end latency/jitter, preprocessing parity, explicit reject handling, and boring operational stuff (logging + safe defaults).
- **Intended runtime**: camera → preprocessing → model → decision → actuation. (Some pieces exist, some don’t — details below.)

---

## Current Implementation (What Exists)

### Offline inference
- **Primary implementation**: `CutleryInferencer` (`src/cutlery_classifier/inference/inferencer.py`)
  - Loads a checkpoint containing `config` + `class_names` + `model_state_dict`.
  - Runs preprocessing + model forward pass.
  - Returns top-k softmax probabilities and `inference_time_ms` (see Performance notes).

### Model + preprocessing
- **Model factory**: `src/cutlery_classifier/models/factory.py`
  - Supports `resnet18` and `mobilenet_v2`.
  - Can adapt pretrained RGB weights to grayscale by reducing/averaging first-layer weights.
- **Transforms**: `src/cutlery_classifier/data/transforms.py`
  - Provides a consistent “base transform” for test/inference (detailed below).

### Evaluation setup
- `scripts/evaluate_on_test_set.py`:
  - Runs file-based inference over a directory-structured test set and writes:
    - `results/evaluation/confusion_matrix.png`
    - `results/evaluation/metrics.json`
- `scripts/test_dataset_inference.py`:
  - Alternative evaluation script that also generates run artifacts in `outputs/run_<timestamp>/`
  - Includes optional confidence analysis and stress tests (offline, not edge-integrated).

---

## Edge Runtime Roadmap (Not Yet Implemented)

To turn this into a real edge system (Pi + camera + sorter), I still need to build:
- **Camera capture loop** (buffering + dropped-frame policy)
- **ONNXRuntime runner** (ARM CPU, same preprocessing as PyTorch)
- **Decision handling** (reject/margin/“manual lane” behavior) wired into runtime
- **Actuator interface** (GPIO/serial/PLC) + safe default state
- **End-to-end latency measurement** (capture→actuate, plus jitter under load)

---

## Preprocessing (Single Source of Truth)

The offline inference path uses the base transforms from `src/cutlery_classifier/data/transforms.py`:

- **Resize**: to `image_size` (default `[320, 320]` from config; commonly 320×320 in this repo)
- **Grayscale**: `num_output_channels=1`
- **ToTensor**: torchvision conversion to `torch.Tensor`
- **Normalize**: custom grayscale normalization using ImageNet-averaged stats
  - **mean**: `0.449`
  - **std**: `0.226`

Pragmatic notes:
- The old README claimed a **center crop to 224×224**. That is **not** in the base transforms.
- There are still scripts with their own transforms; before deploying, this needs to collapse to one canonical preprocessing path (PyTorch + ONNXRuntime).

---

## Decision / Reject Handling (Planned)

In production, `argmax` is not enough. This needs explicit reject handling, e.g.:
- reject/unknown when confidence is low
- margin check when top-1 and top-2 are too close
- simple temporal confirmation across frames before actuating

Status in this repo:
- The main runtime inference returns probabilities but **does not apply a reject policy**.
- There is analysis-oriented code (e.g., confidence statistics) in evaluation modules, but it is **not wired** into a deployable runtime loop.

---

## Performance

### What is measured today
- `CutleryInferencer.predict()` returns `inference_time_ms` measured around preprocessing + forward pass + postprocessing in Python.

### What is *not* measured yet (edge-relevant)
- **End-to-end** timing including camera capture and actuation.
- **Raspberry Pi CPU** benchmarks.
- **ONNXRuntime** inference latency on ARM CPU.
- **Deterministic measurement method** (e.g., CUDA synchronization for GPU measurements).

### Raspberry Pi targets (placeholders for real measurements)
- **Model-only ONNXRuntime latency (ms)**: TBD
- **Preprocess latency (ms)**: TBD
- **End-to-end (capture→actuate) latency (ms)**: TBD
- **Sustained FPS (thermal steady state)**: TBD

---

## Reproducible Run

### Setup

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

pip install -e .
```

### Offline single-image inference (PyTorch checkpoint)

This uses the packaged entrypoint that forwards to `scripts/run_inference.py`:

```bash
cutlery-inference --device cpu --image <path/to/image.jpg> --model <path/to/checkpoint.pth>
```

Notes:
- A compatible checkpoint must contain `config`, `class_names`, and `model_state_dict` (see `CutleryInferencer.load_model()`).
- If you don’t have the checkpoint and dataset folders locally, the commands will fail (this repo does not guarantee those artifacts are present).

### Offline evaluation on a directory-structured test set

```bash
python scripts/evaluate_on_test_set.py \
  --device cpu \
  --model <path/to/checkpoint.pth> \
  --test-dir <path/to/test_dir>
```

Outputs (relative to repo root):
- `results/evaluation/metrics.json`
- `results/evaluation/confusion_matrix.png`

---

## Training Notes (short)

Training is implemented but not the focus of this README.
- Config examples live in `config/` (e.g., `config/train_config.yaml`).
- The training pipeline code is under `src/cutlery_classifier/training/`.
- Hyperparameter tuning scripts exist under `scripts/` (e.g., Optuna-related tooling).

For a portfolio-quality edge story, training details should move into `docs/training.md` and the README should stay focused on **system behavior and deployability**.

---

## Known Gaps / Risks

- **Missing camera loop**
- **Missing actuator integration**
- **Reject handling not wired into runtime**
- **Preprocessing parity risk** (multiple transform definitions in scripts)
- **Dependency bloat** (dev tooling mixed into core requirements)

---

## Next Engineering Steps

To turn this into a real edge system, the next steps are:
- Add a minimal runtime skeleton (camera loop + runner + policy + actuator interfaces).
- Implement ONNXRuntime inference on ARM and lock preprocessing parity against the PyTorch path.
- Wire in reject handling and log “why” (confidence, margin, etc.) for every decision.
- Build an end-to-end latency/jitter harness for Pi (including thermal steady state).
- Split runtime vs dev dependencies so a Pi install is small and predictable.

---

## License

MIT License (see `LICENSE`).
