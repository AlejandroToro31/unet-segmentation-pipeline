import os
import io
import base64
import logging
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import segmentation_models_pytorch as smp

# --- LOGGING INFRASTRUCTURE ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [UNET-API] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("UNetAPI")

# --- GLOBAL STATE & CONFIGURATION ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_unet_bottle.pth")
DEFECT_AREA_THRESHOLD = int(os.getenv("DEFECT_AREA_THRESHOLD", "50"))
ml_state = {} 

# --- THE PREPROCESSING PIPELINE ---
image_transforms = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- SERVER LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting Semantic Segmentation API...")
    logger.info(f"Allocating U-Net (ResNet34 Backbone) to: {DEVICE.type.upper()}")
    
    try:
        # Reconstruct the exact SMP architecture from the R&D phase
        model = smp.Unet(
            encoder_name="resnet34", 
            encoder_weights=None, # We don't need ImageNet weights for inference
            in_channels=3, 
            classes=1, 
            activation=None
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval() 
        
        ml_state["model"] = model
        logger.info(f"Neural Network successfully locked in VRAM. Pixel Area Threshold: {DEFECT_AREA_THRESHOLD}")
    except Exception as e:
        logger.critical(f"Failed to load model artifact: {e}")
        raise RuntimeError("Server boot aborted due to missing artifact.")
        
    yield 
    
    logger.info("Shutting down. Clearing VRAM...")
    ml_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- API INSTANTIATION ---
app = FastAPI(
    title="Semantic Anomaly Segmentation API",
    description="U-Net microservice for pixel-level industrial defect localization.",
    version="1.0.0",
    lifespan=lifespan
)

# --- RESPONSE SCHEMA ---
class SegmentationResponse(BaseModel):
    filename: str
    is_defect: bool
    defect_pixel_count: int
    threshold: int
    mask_base64: str | None # The actual image data encoded as text

# --- HELPER FUNCTION: BASE64 ENCODING ---
def encode_mask_to_base64(binary_mask: np.ndarray) -> str:
    """Converts a (256, 256) binary numpy array into a base64 PNG string."""
    # Convert 0/1 to 0/255 for standard image rendering
    visual_mask = (binary_mask * 255).astype(np.uint8)
    
    # Encode as PNG in memory
    success, encoded_image = cv2.imencode('.png', visual_mask)
    if not success:
        raise ValueError("Failed to encode mask to PNG format.")
        
    return base64.b64encode(encoded_image).decode('utf-8')

# --- THE ENDPOINT ---
@app.post("/segment/", response_model=SegmentationResponse)
async def segment_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid payload. Image required.")

    try:
        # 1. Direct Memory Decode
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("OpenCV failed to decode image.")

        # 2. R&D Pipeline Alignment
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = image_transforms(image=img)['image'].unsqueeze(0).to(DEVICE)
        model = ml_state["model"]
        
        # 3. Forward Pass & Binarization
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).squeeze().cpu().numpy()
            
        # 4. Pixel Analytics
        defect_pixels = int(np.sum(pred_mask))
        is_defect = defect_pixels > DEFECT_AREA_THRESHOLD
        
        # 5. Visual Encoding (Only send the heavy Base64 string if a defect is actually found)
        mask_b64 = encode_mask_to_base64(pred_mask) if is_defect else None

        logger.info(f"Segmented '{file.filename}': Defect={is_defect} | Pixels={defect_pixels}")

        return SegmentationResponse(
            filename=file.filename,
            is_defect=is_defect,
            defect_pixel_count=defect_pixels,
            threshold=DEFECT_AREA_THRESHOLD,
            mask_base64=mask_b64
        )

    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during segmentation.")