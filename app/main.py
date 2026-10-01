import io
import logging
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.config import DEFAULT_CHECKPOINT, ModelConfig
from src.dataset import build_eval_transform
from src.model import load_checkpoint
from src.utils import setup_logging

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

logger = setup_logging(Path("artifacts/logs"), "api")

app = FastAPI(
    title="Visual Defect Detection API",
    version="1.0.0",
    description="Production-style Normal vs Defective image classification API.",
)

_model = None
_transform = build_eval_transform(ModelConfig.image_size)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str


def get_model():
    global _model
    if _model is None:
        if not DEFAULT_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {DEFAULT_CHECKPOINT}. "
                "Train the model first."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _ = load_checkpoint(DEFAULT_CHECKPOINT, device=device)
        _model.to(device).eval()
        _model._device = device
        logger.info("Model loaded: %s on %s", ModelConfig.model_name, device)
    return _model


@app.get("/health")
def health():
    return {
        "status": "ok" if DEFAULT_CHECKPOINT.exists() else "degraded",
        "model_available": DEFAULT_CHECKPOINT.exists(),
        "model_version": ModelConfig.model_name,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES} byte upload limit.",
        )

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.") from exc
    except Exception as exc:
        logger.exception("Image decoding failed.")
        raise HTTPException(status_code=400, detail="Unable to decode image.") from exc

    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    tensor = _transform(image).unsqueeze(0).to(model._device)

    try:
        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            pred_index = int(torch.argmax(probabilities).item())
    except Exception as exc:
        logger.exception("Inference failed.")
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    classes = ModelConfig.class_names
    probs = {
        classes[0]: round(float(probabilities[0].item()), 4),
        classes[1]: round(float(probabilities[1].item()), 4),
    }

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    result = PredictionResponse(
        predicted_class=classes[pred_index],
        confidence=round(float(probabilities[pred_index].item()), 4),
        probabilities=probs,
        latency_ms=latency_ms,
        model_version=ModelConfig.model_name,
    )

    logger.info(
        "prediction=%s confidence=%.4f latency_ms=%.2f filename=%s",
        result.predicted_class,
        result.confidence,
        result.latency_ms,
        file.filename,
    )
    return result
