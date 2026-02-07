## Cutlery Classifier MVP — README v2 (systems-first)

An **offline inference MVP** for a cutlery type classifier (fork/knife/spoon), packaged as a Python library + scripts. This repo is written as an **Edge AI Systems Engineering case study** with a clear roadmap toward a Raspberry Pi + camera deployment.

### Executive Summary (1 screen)
- **What the system does**: classifies a single image of cutlery into one of three classes: `fork`, `knife`, `spoon`.
- **Current MVP scope**: offline, file-based inference + offline test-set evaluation + model export to ONNX/TorchScript (export only).
- **Target hardware (roadmap)**: Raspberry Pi (CPU) + camera, with **ONNX Runtime** as the baseline inference engine.
- **Key constraints (edge)**:
  - **Latency**: end-to-end (capture → preprocess → inference → decision → actuate) must be measured, not assumed.
  - **Preprocessing parity**: training/inference transforms must be identical across PyTorch and ONNXRuntime.
  - **Decision robustness**: edge deployments need an explicit reject/unknown policy, not just `argmax`.
  - **Operational readiness**: logging, watchdog behavior, and safe actuator defaults are required for credible industrial stories.

---

## System Architecture

The intended deployment pipeline is:

```
              (PLANNED)                 (EXISTS)                 (PLANNED)
Camera/Driver ────────> Preprocess ────> Model Runner ────> Decision Policy ────> Actuation
   [missing]            [exists for files]  [PyTorch exists]      [not wired]       [missing]
                                              [ONNX export only]
```

### Dataflow: Camera → Preprocess → Model → Decision → Actuation
- **Camera**: not implemented (no capture loop / camera driver integration in repo).
- **Preprocess**: implemented for offline image files (torchvision transforms).
- **Model**: implemented for PyTorch checkpoint inference; ONNX export exists (runtime runner not implemented).
- **Decision**: planned (reject/margin policy not integrated into runtime inference).
- **Actuation**: not implemented (no GPIO/PLC/serial interface).

---

## Current Implementation (What Exists)

### Offline inference
- **Primary implementation**: `CutleryInferencer` in `src/cutlery_classifier/inference/inferencer.py`
  - Loads a checkpoint containing `config` + `class_names` + `model_state_dict`.
  - Runs preprocessing + model forward pass.
  - Returns top-k softmax probabilities and a measured `inference_time_ms` (see Performance notes).

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

To make this an actual edge system (Pi + camera + actuator), the repo still needs:
- **Camera capture loop**
  - A frame acquisition module (e.g., V4L2/libcamera/OpenCV) with buffering and dropped-frame policy.
- **ONNXRuntime runner**
  - A runtime that loads the exported `.onnx` model and matches PyTorch preprocessing exactly.
  - Threading controls and benchmarking on ARM CPU.
- **Decision policy integration (reject / margin)**
  - Runtime wiring of confidence/margin gating and a clear “reject” output behavior.
- **Actuator interface**
  - A hardware abstraction for “sort left/right/reject” (GPIO/serial/PLC), including a safe default.
- **End-to-end latency measurement**
  - Measure: capture time + preprocess + inference + decision + actuator command (and jitter under load).

---

## Preprocessing (Single Source of Truth)

The offline inference path uses the base transforms from `src/cutlery_classifier/data/transforms.py`:

- **Resize**: to `image_size` (default `[320, 320]` from config; commonly 320×320 in this repo)
- **Grayscale**: `num_output_channels=1`
- **ToTensor**: torchvision conversion to `torch.Tensor`
- **Normalize**: custom grayscale normalization using ImageNet-averaged stats
  - **mean**: `0.449`
  - **std**: `0.226`

Important notes:
- The current README v1 stated a **center crop to 224×224**. That is **not implemented** in the base transforms.
- There are multiple scripts with their own transforms; for deployment credibility, the project should converge on **one** preprocessing definition and reuse it everywhere (PyTorch + ONNXRuntime).

---

## Decision Policy (Planned)

Edge deployments usually require a decision policy beyond `argmax`, for example:
- **Reject / unknown** when top-1 confidence is below a threshold
- **Margin gating** when top-1 minus top-2 is too small (ambiguous)
- **Temporal confirmation** across multiple frames before actuating

Status in this repo:
- The main runtime inference returns probabilities but **does not apply a reject policy**.
- There is analysis-oriented code (e.g., confidence statistics) in evaluation modules, but it is **not wired** into a deployable runtime loop.

---

## Performance

### What is measured today
- `CutleryInferencer.predict()` returns `inference_time_ms` measured around:
  - preprocessing + model forward pass + postprocessing in Python.

### What is *not* measured yet (edge-relevant)
- **End-to-end** timing including camera capture and actuation.
- **Raspberry Pi CPU** benchmarks.
- **ONNXRuntime** inference latency on ARM CPU.
- **Deterministic measurement method** (e.g., CUDA synchronization for GPU measurements).

### Target platform placeholders (to be filled with real numbers)
| Metric | Raspberry Pi (CPU) | Notes |
|---|---:|---|
| Model-only inference latency (ms) | TBD | ONNXRuntime, batch=1 |
| Preprocess latency (ms) | TBD | resize + grayscale + normalize |
| End-to-end latency (ms) | TBD | capture→actuate |
| Sustained FPS @ thermal steady state | TBD | includes throttling |

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

- **Missing camera loop**: no capture pipeline, buffering strategy, or camera configuration.
- **Missing actuator integration**: no interface for a sorter mechanism (GPIO/PLC/serial) and no safe-state behavior.
- **Reject policy not integrated**: probabilities are produced but not converted into an auditable accept/reject decision.
- **Preprocessing parity risk**: multiple scripts define transforms; edge/ONNX must match the single canonical preprocessing.
- **Dependency bloat for edge**: dev tooling (plots/Grad-CAM/Optuna) is mixed into core install requirements; edge deploy should be slim.
- **Packaging inconsistencies**: multiple dependency specs and versions (`requirements.txt`, `pyproject.toml`, `setup.py`) disagree.
- **Reproducibility risk**: tests and scripts reference data/model artifacts that may not be in the repo by default.

---

## Next Engineering Steps

Concrete TODOs to turn this into an edge deployment:
- Define a deployable module boundary: `camera`, `preprocess`, `runner`, `policy`, `actuator`, `app`.
- Add an ONNXRuntime inference runner and verify preprocessing parity vs PyTorch on a fixed test vector.
- Implement a decision policy (confidence + margin + temporal voting) and log decisions with reasons.
- Add an end-to-end benchmark harness for Pi CPU (including thermal steady state and jitter).
- Create an edge-oriented dependency profile (minimal runtime requirements).
- Add a system-level runbook: “bring-up on Pi”, service install, logging locations, and health checks.

---

## License

MIT License (see `LICENSE`).
