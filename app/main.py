"""
Semantic Anomaly Segmentation API — UNet Defect Localization
================================================================
Production FastAPI microservice for pixel-wise defect segmentation
on MVTec AD bottle images.

CRITICAL — Scoring Methodology Alignment:
    Defect pixel counting MUST include the same morphological opening
    post-processing used in the training/calibration script. A
    threshold calibrated against cleaned pixel counts is meaningless
    if the API counts raw, noisier pixels. This file mirrors the
    cleanup step exactly from the training pipeline.

Features:
    - asyncio.to_thread for non-blocking inference
    - Morphological opening noise cleanup (matches training calibration)
    - Conditional Base64 mask encoding — only sent when a defect is found
    - Model warmup on startup

Endpoints:
    GET  /              → API metadata
    GET  /health        → Liveness check
    GET  /ready         → Readiness check (model loaded)
    POST /api/v1/segment → Defect segmentation inference

Environment Variables:
    MODEL_PATH            : Path to best_unet_bottle.pth
    DEFECT_AREA_THRESHOLD : Fallback threshold (default: 50 pixels)
                            Recommended: use the value from
                            calibrate_threshold() in the training script.

Known Improvement (not yet implemented):
    The UNet architecture instantiation is duplicated between this file
    and the training script. Extracting model construction into a
    shared model.py would prevent architecture drift.
"""

# ── Standard Library
import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

# ── Third Party
import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


# ════════════════════════════════════════════════════════
# 1. LOGGING INFRASTRUCTURE
# ════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [UNet-API] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UNetAPI")


# ════════════════════════════════════════════════════════
# 2. GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_unet_bottle.pth")

# Fallback default — should be overridden with the value returned by
# calibrate_threshold() in the training script for production accuracy.
DEFECT_AREA_THRESHOLD: int = int(os.getenv("DEFECT_AREA_THRESHOLD", "50"))

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Singleton store — model loaded once at startup
ml_state: Dict = {}


# ── Preprocessing Pipeline
# CRITICAL: must match _build_transform(train=False) in training script.
image_transforms = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ════════════════════════════════════════════════════════
# 3. SEGMENTATION HELPERS — MUST MATCH TRAINING SCRIPT
# ════════════════════════════════════════════════════════

def run_segmentation(
    model: nn.Module,
    input_tensor: torch.Tensor
) -> np.ndarray:
    """
    Synchronous inference function — runs in thread pool via asyncio.to_thread.

    Returns the CLEANED binary mask (post morphological opening) —
    must mirror predict_segmentation() in the training script exactly.
    """
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # Morphological opening — erosion then dilation, removes small noisy
    # false-positive blobs while preserving genuine structural defects.
    # MUST match training script's predict_segmentation() exactly —
    # a threshold calibrated on cleaned counts is meaningless otherwise.
    kernel = np.ones((3, 3), np.uint8)
    pred_mask_clean = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

    return pred_mask_clean


def encode_mask_to_base64(binary_mask: np.ndarray) -> str:
    """Converts a binary numpy mask into a base64-encoded PNG string."""
    visual_mask = (binary_mask * 255).astype(np.uint8)
    success, encoded_image = cv2.imencode(".png", visual_mask)
    if not success:
        raise ValueError("Failed to encode mask to PNG format.")
    return base64.b64encode(encoded_image).decode("utf-8")


# ════════════════════════════════════════════════════════
# 4. SERVER LIFESPAN — MODEL SINGLETON PATTERN
# ════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager — controls UNet model lifecycle.

    STARTUP: Loads model (no ImageNet weights needed — custom state_dict
             is loaded instead), runs warmup inference.
    SHUTDOWN: Clears model state, releases GPU memory.
    """
    logger.info("Booting Semantic Segmentation API...")
    logger.info(f"Device: {DEVICE.type.upper()} | Threshold: {DEFECT_AREA_THRESHOLD}px")

    try:
        # encoder_weights=None — we load our own fine-tuned state_dict below,
        # no need to download ImageNet weights for inference
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        ml_state["model"] = model
        logger.info("UNet (ResNet34) loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifact: {e}")
        raise RuntimeError(
            f"Server boot aborted — artifact not found at: {MODEL_PATH}"
        ) from e

    # ── Warmup inference — compiles CUDA kernels before first real request
    logger.info("Running warmup inference...")
    dummy = torch.zeros(1, 3, 256, 256).to(DEVICE)
    with torch.inference_mode():
        ml_state["model"](dummy)
    logger.info("Model warmed up. API ready to serve requests.")

    yield

    logger.info("Shutting down. Releasing resources...")
    ml_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete.")


# ════════════════════════════════════════════════════════
# 5. API INSTANTIATION
# ════════════════════════════════════════════════════════

app = FastAPI(
    title="Semantic Anomaly Segmentation API",
    description=(
        "UNet (ResNet34) microservice for pixel-level industrial defect "
        "localization on MVTec AD bottle images."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════
# 6. RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════

class SegmentationResponse(BaseModel):
    """Inference result payload."""
    filename            : str
    is_defect           : bool
    defect_pixel_count   : int
    threshold            : int
    mask_base64          : Optional[str] = None  # None when no defect detected
    latency_ms           : float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status      : str
    model_path  : str
    threshold   : int


# ════════════════════════════════════════════════════════
# 7. UTILITY ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/", tags=["Utility"])
async def root() -> dict:
    """API metadata — entry point for documentation discovery."""
    return {
        "api"    : "Semantic Anomaly Segmentation API",
        "version": "1.1.0",
        "docs"   : "/docs",
        "health" : "/health",
        "ready"  : "/ready",
        "segment": "/api/v1/segment",
    }


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    """Liveness check — confirms the API process is running."""
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadyResponse, tags=["Utility"])
async def ready() -> ReadyResponse:
    """Readiness check — confirms the model is loaded and inference is possible."""
    if ml_state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be initializing."
        )
    return ReadyResponse(
        status   ="ready",
        model_path=MODEL_PATH,
        threshold =DEFECT_AREA_THRESHOLD,
    )


# ════════════════════════════════════════════════════════
# 8. INFERENCE ENDPOINT
# ════════════════════════════════════════════════════════

@app.post("/api/v1/segment", response_model=SegmentationResponse, tags=["Inference"])
async def segment_image(file: UploadFile = File(...)) -> SegmentationResponse:
    """
    Defect segmentation endpoint.

    Accepts an image upload, runs UNet inference to produce a pixel-wise
    defect mask, cleans noise via morphological opening, and returns a
    structured response — including the Base64-encoded mask only when
    a defect is actually detected (bandwidth optimization).

    Pipeline:
        1. MIME type validation       → reject non-image content types
        2. Payload size validation    → reject files > 10MB
        3. In-memory image decoding   → zero disk I/O via cv2.imdecode
        4. BGR → RGB conversion       → OpenCV reads BGR, model expects RGB
        5. Thread-offloaded inference → asyncio.to_thread (non-blocking)
        6. Morphological cleanup      → matches training calibration exactly
        7. Conditional Base64 encode  → only when is_defect=True
        8. Structured response        → Pydantic-validated JSON payload

    Raises:
        400 : Invalid content type (non-image upload)
        413 : Payload exceeds 10MB limit
        422 : Valid image type but content cannot be decoded (corrupted)
        500 : Unexpected inference error
        503 : Model not loaded
    """

    # ── Model availability check
    model = ml_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Server may still be initializing."
        )

    # ── Step 1: MIME type validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid content type: '{file.content_type}'. "
                "Only image/jpeg, image/png, image/webp accepted."
            )
        )

    # ── Step 2: Read bytes + size validation
    image_bytes: bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Payload too large: {len(image_bytes) / 1024 / 1024:.1f}MB. "
                f"Maximum allowed: {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB."
            )
        )

    try:
        inference_start = time.perf_counter()

        # ── Step 3: In-memory image decoding (zero disk I/O)
        np_arr: np.ndarray = np.frombuffer(image_bytes, np.uint8)
        img: Optional[np.ndarray] = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Image decoding failed. File may be corrupted, truncated, "
                    "or content type was incorrectly declared."
                )
            )

        # ── Step 4: BGR → RGB conversion
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Step 5: Preprocessing + thread-offloaded inference
        input_tensor = image_transforms(image=img)["image"].unsqueeze(0).to(DEVICE)

        # asyncio.to_thread offloads CPU/GPU-bound inference to a thread pool.
        # Event loop stays free to accept other requests during computation.
        pred_mask_clean = await asyncio.to_thread(
            run_segmentation, model, input_tensor
        )

        # ── Step 6: Pixel analytics on the CLEANED mask
        defect_pixels = int(np.sum(pred_mask_clean))
        is_defect = defect_pixels > DEFECT_AREA_THRESHOLD

        # ── Step 7: Conditional Base64 encoding — bandwidth optimization
        # Only attach the heavy mask payload when a defect is actually found
        mask_b64 = encode_mask_to_base64(pred_mask_clean) if is_defect else None

        latency_ms = round((time.perf_counter() - inference_start) * 1000, 2)

        logger.info(
            f"Segmented '{file.filename or 'unknown'}' | "
            f"Defect: {is_defect} | Pixels: {defect_pixels} | "
            f"Latency: {latency_ms}ms"
        )

        return SegmentationResponse(
            filename           =file.filename or "unknown",
            is_defect          =is_defect,
            defect_pixel_count =defect_pixels,
            threshold          =DEFECT_AREA_THRESHOLD,
            mask_base64        =mask_b64,
            latency_ms         =latency_ms,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected inference error on '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during segmentation. Check server logs."
        )

    finally:
        try:
            del image_bytes, np_arr, img, input_tensor
        except NameError:
            pass