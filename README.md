# UNet Defect Segmentation API

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

A production-deployed pixel-wise semantic segmentation microservice for industrial defect localization on MVTec AD bottle images. Instead of simply classifying an image as defective, this system maps the exact spatial boundaries of the anomaly and returns a compressed Base64 mask for client-side overlay rendering.

> **Part of a three-project comparison:** Together with the [Supervised Classifier](#) and [Unsupervised Anomaly Detector](#), this project completes a full exploration of defect detection paradigms on the same MVTec bottle dataset — classification (what), unsupervised anomaly detection (is it normal), and segmentation (exactly where).

---

## System Architecture

| Component | Implementation | Details |
|-----------|---------------|---------|
| **Segmentation Engine** | UNet (segmentation_models_pytorch) | ResNet34 encoder, ImageNet pretrained, fully fine-tuned |
| **Training Loss** | Compound Loss | 0.5 × BCEWithLogits + 0.5 × Dice — combats extreme class imbalance |
| **Inference Output** | Raw logits → Sigmoid → Threshold | activation=None for numerical stability with BCEWithLogitsLoss |
| **Noise Cleanup** | Morphological opening | Removes small false-positive speckle before pixel counting |
| **Web Framework** | FastAPI + Uvicorn | ASGI, async request handling |
| **Image Pipeline** | Albumentations + cv2.imdecode | Zero disk I/O, synchronized image+mask transforms |
| **Container** | python:3.10-slim | Non-root user, layer-cached builds, HEALTHCHECK |

**Why Compound Loss matters:**
Industrial defects often occupy less than 1% of the image. BCE alone causes the model to predict all-zero masks while achieving deceptively high pixel accuracy. Dice Loss forces spatial localization regardless of defect size by ignoring the dominant background class entirely; BCE provides stable gradients early in training when Dice's overlap term is near zero. Combined, they solve each other's failure modes.

**Negative sampling:**
Nominal ("good") images are included in training with zero-tensor masks. Without this, the model develops a bias toward assuming every image contains a defect somewhere, causing false-positive hallucinations on perfectly normal bottles in production.

**Bandwidth optimization:**
The heavy Base64-encoded mask is only attached to the response when a defect actually breaches the threshold — sending a mostly-black mask for every perfect bottle would waste bandwidth at scale.

---

## Tech Stack

- **Deep Learning:** PyTorch 2.1, segmentation-models-pytorch
- **Web Server:** FastAPI 0.104, Uvicorn (with uvloop + httptools)
- **Computer Vision:** OpenCV (`opencv-python-headless`), Albumentations
- **DevOps:** Docker, python:3.10-slim base image

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API metadata and endpoint discovery |
| `GET` | `/health` | Liveness check — is the process running? |
| `GET` | `/ready` | Readiness check — is the model loaded? |
| `POST` | `/api/v1/segment` | Defect segmentation inference |

---

## Project Structure

```
unet-bottle-segmentation/
├── app/
│   └── main.py                          # FastAPI inference endpoint
├── models/
│   └── best_unet_bottle.pth             # Model artifact (download separately)
├── training/
│   └── UNet_Segmentation_defects.ipynb  # Training, calibration & evaluation pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Download the Model Artifact

Due to GitHub file size limits, trained weights are stored externally.

1. Download `best_unet_bottle.pth` from: [Model Registry (Google Drive)](https://drive.google.com/file/d/1jRp_T8NHWgksMLqPqvSAfUuQhHjZYw7l/view?usp=sharing)
2. Place it inside the `models/` directory:

```
models/
└── best_unet_bottle.pth
```

### 2. Build the Container

```bash
docker build -t unet-segmentation-api:v1 .
```

### 3. Run the Container

Default configuration (fallback threshold — 50 pixels):
```bash
docker run -p 8000:8000 unet-segmentation-api:v1
```

With your calibrated threshold (recommended — see Training Pipeline below):
```bash
docker run -p 8000:8000 \
  -e DEFECT_AREA_THRESHOLD=73 \
  unet-segmentation-api:v1
```

Verify the API is ready:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### 4. Run Inference

Navigate to **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

Upload an image to `POST /api/v1/segment` and inspect the JSON response.

---

## Example Response

**Defect detected:**
```json
{
  "filename": "bottle_broken_large_007.png",
  "is_defect": true,
  "defect_pixel_count": 312,
  "threshold": 50,
  "mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "latency_ms": 34.7
}
```

**No defect (mask omitted — bandwidth optimization):**
```json
{
  "filename": "bottle_good_021.png",
  "is_defect": false,
  "defect_pixel_count": 4,
  "threshold": 50,
  "mask_base64": null,
  "latency_ms": 29.1
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|--------------|
| `MODEL_PATH` | `models/best_unet_bottle.pth` | Path to trained UNet artifact |
| `DEFECT_AREA_THRESHOLD` | `50` | Fallback threshold (pixels) — override with calibrated value |

> **Important:** The default `50` is an uncalibrated fallback. The recommended threshold is derived empirically via `calibrate_threshold()` in the training pipeline, which measures false-positive pixel counts on a held-out nominal validation subset.

---

## Training Pipeline

To retrain, calibrate, or benchmark, open `training/UNet_Segmentation_defects.ipynb`:

```bash
pip install jupyterlab segmentation-models-pytorch albumentations scikit-learn matplotlib tqdm
jupyter lab
```

The notebook handles:
1. MVTec AD dataset download with **dynamic defect category detection** — generalizes beyond the bottle category
2. UNet training via Compound Loss (BCE + Dice), checkpointing on best validation IoU
3. **Threshold calibration** — percentile-based, derived from a held-out nominal-only validation subset
4. **Full test-set evaluation** — IoU and Dice, overall and per defect category
5. Visual inspection grid: prediction overlay with per-sample IoU

---

## Edge Case Behavior — Domain Shift

This model operates strictly within the structural constraints of the MVTec bottle spatial manifold it was trained on:

- Non-bottle objects → unreliable mask predictions
- Severe lighting changes outside training distribution → degraded IoU
- Non-standard camera angles → feature distribution shift

**This system is designed strictly for fixed-camera, controlled-lighting industrial environments matching the original MVTec AD acquisition setup.**

---

## Docker Notes

**OS dependencies:** `python:3.10-slim` strips many system libraries. `libgl1` and `libglib2.0-0` are reinstalled — required by OpenCV's image processing backend even with the headless build.

**Non-root execution:** Container runs as `api_user` — principle of least privilege.

**Health monitoring:** Docker's native `HEALTHCHECK` polls `/health` every 30 seconds with a 60-second startup grace period for model loading.