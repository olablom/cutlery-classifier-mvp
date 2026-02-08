# Cutlery Classifier MVP

Offline inference MVP for classifying cutlery (`fork`, `knife`, `spoon`), written as an edge deployment case study. The point of this repo is to be honest about what’s here today and what’s still missing before it can run on a Raspberry Pi with a camera.

## Overview
- What exists: file-based inference + offline evaluation scripts.
- What doesn’t: camera loop, ONNXRuntime runner, actuator integration.
- Target (roadmap): Raspberry Pi (CPU) with ONNX Runtime.
- Constraints: end-to-end latency/jitter, preprocessing parity, reject handling, and reliable logging/safe defaults.

Intended runtime: camera → preprocessing → model → decision → actuation.

## Model

This MVP is **checkpoint-driven**. The runtime loads a PyTorch checkpoint and runs file-based inference.

Verified from local checkpoints used during development:
- Backbone family: **ResNet**
- Exact variant: **ResNet18 (BasicBlock)**

Note on checkpoint formats (both exist locally):
- Some checkpoints are full dicts and include `config`, `class_names`, and `model_state_dict` (e.g. `type_detector_best.pth`).
- Some checkpoints are **state_dict only** (`OrderedDict`) and do not include metadata (e.g. `type_detector_best_tuned.pth`).

## Preprocessing

Current preprocessing is configured for grayscale + fixed resize. See `config/train_config.yaml`:
- grayscale input
- resize to 320×320
- normalize with mean `0.449`, std `0.226`

## Decision / reject handling (planned)

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
- The runtime expects either:
  - a full checkpoint dict containing `model_state_dict` (and optionally `config` / `class_names`), or
  - a raw `state_dict` (`OrderedDict`) without metadata.
- Local artifacts such as checkpoints/exports may exist under `models/` but are **ignored by git** and are not guaranteed to be present in a fresh clone.

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

Local-only artifact outputs:
- Some scripts may write to `outputs/` and `results/` (both ignored by git).

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
