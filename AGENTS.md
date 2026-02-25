# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

SafeCommunity-AI is a Python-based intelligent community video surveillance system with two sub-modules:

- **`yolo/`** — YOLOv8 PyTorch object detection (80 COCO classes + custom smoke/fire). Run from `yolo/` with `sys.path` including `yolo/`.
- **`lab_anomaly/`** — Video anomaly detection using VideoMAE ViT + MIL classifier + KMeans/One-Class SVM. Run as a Python package from the repo root (e.g. `python -m lab_anomaly.train.extract_embeddings`).

### Known Issues

- `yolo/` is missing several utility files (`utils/utils_bbox.py`, `utils/utils.py`, `utils/utils_map.py`, `utils/utils_fit.py`, and top-level `yolo.py` YOLO class). The backbone (`nets/backbone.py`) and detection head (`nets/yolo.py`) work in isolation but full training/inference pipelines cannot run.
- `lab_anomaly/models/ranking_loss.py` is missing, so `train_known_classifier.py` cannot be imported. All other modules work.
- `yolo/requirements.txt` pins very old versions (e.g. `scipy==1.2.1`, `numpy==1.17.0`) incompatible with Python 3.12. Install modern versions without pins.

### Dependencies

Both modules share a common set of Python packages. Install CPU-only PyTorch (no GPU available in cloud) plus the shared dependencies:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy tqdm transformers PyYAML scikit-learn joblib scikit-image tensorboard scipy matplotlib h5py Pillow flake8
```

### Running & Testing

- **Lint**: `python -m flake8 lab_anomaly/ yolo/ --max-line-length=120 --select=E9,F63,F7,F82`
- **YOLO backbone test**: `cd /workspace && python3 -c "import sys; sys.path.insert(0,'yolo'); from nets.backbone import Backbone; ..."` (see README for full examples)
- **lab_anomaly models**: Import and run from repo root, e.g. `python -c "from lab_anomaly.models.mil_head import MILClassifier, MILHeadConfig; ..."`
- **lab_anomaly training pipeline**: Requires dataset + labels in `lab_dataset/`. See `lab_anomaly/readme.txt` for step-by-step.
- **RTSP inference**: Requires a live RTSP camera feed + trained model checkpoints. See `lab_anomaly/configs/rtsp_service_example.yaml`.

### Caveats

- No CUDA GPU in cloud VM — PyTorch runs on CPU only. Both modules auto-detect and fall back to CPU.
- No pre-trained weights or datasets are included (copyright-restricted per README). Full training/inference requires external data.
- The HuggingFace `MCG-NJU/videomae-base` model downloads on first use (~350MB). In offline environments, pre-cache it.
