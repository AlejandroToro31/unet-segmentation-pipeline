# ==============================================================
# UNet Segmentation API — Production Dockerfile
# ==============================================================
# Pixel-wise defect localization microservice.
# CPU inference, non-root execution.
#
# Build:
#   docker build -t unet-segmentation-api:v1 .
#
# Run:
#   docker run -p 8000:8000 unet-segmentation-api:v1
#
# Run with calibrated threshold override:
#   docker run -p 8000:8000 \
#     -e DEFECT_AREA_THRESHOLD=73 \
#     unet-segmentation-api:v1
# ==============================================================

# ── Base Image
FROM mirror.gcr.io/library/python:3.10-slim

# ── Image Metadata
LABEL version="1.1.0"
LABEL description="UNet Segmentation API — ResNet34 encoder, pixel-wise defect localization"

# ── Python Runtime Configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Application Configuration
# DEFECT_AREA_THRESHOLD default is a fallback (50px).
# IMPORTANT: override with the value from calibrate_threshold()
# in the training script before real production deployment.
ENV MODEL_PATH=models/best_unet_bottle.pth \
    DEFECT_AREA_THRESHOLD=50

# ── OS-Level Dependencies
# OpenCV requires libgl1 and libglib2.0-0 on python:3.10-slim,
# even with opencv-python-headless — confirmed via prior debugging.
# curl: required for Docker HEALTHCHECK instruction.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Security: Non-Root User
RUN groupadd -r api_user && useradd -m -r -g api_user api_user

# ── Working Directory
WORKDIR /workspace
RUN chown api_user:api_user /workspace

# ── Layer Caching Strategy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code
COPY --chown=api_user:api_user app/ app/
COPY --chown=api_user:api_user models/ models/

# ── Drop Privileges
USER api_user

# ── Port Declaration
EXPOSE 8000

# ── Health Monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
