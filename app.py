import io
import os
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoint"))
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.70"))

app = FastAPI(title="Image Moderation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModerationResult(BaseModel):
    filename: str
    label: str
    nsfwScore: float
    isSensitive: bool


class ModerationResponse(BaseModel):
    success: bool
    threshold: float
    isSensitive: bool
    flags: List[str]
    results: List[ModerationResult]


def _load_model():
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
    model.eval()
    return processor, model


try:
    IMAGE_PROCESSOR, IMAGE_MODEL = _load_model()
except Exception as exc:
    IMAGE_PROCESSOR = None
    IMAGE_MODEL = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None


@app.get("/health")
def health_check():
    return {
        "ok": IMAGE_MODEL is not None,
        "checkpointDir": CHECKPOINT_DIR,
        "error": MODEL_LOAD_ERROR,
    }


@app.post("/moderate-images", response_model=ModerationResponse)
async def moderate_images(
    files: List[UploadFile] = File(...),
    threshold: float = Form(default=NSFW_THRESHOLD),
):
    if IMAGE_MODEL is None or IMAGE_PROCESSOR is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {MODEL_LOAD_ERROR}")

    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    nsfw_idx = None
    for idx, label in IMAGE_MODEL.config.id2label.items():
        if str(label).strip().lower() == "nsfw":
            nsfw_idx = int(idx)
            break

    if nsfw_idx is None:
        raise HTTPException(status_code=500, detail="NSFW label is missing in model config")

    results: List[ModerationResult] = []

    for upload in files:
        try:
            content = await upload.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}")

        inputs = IMAGE_PROCESSOR(images=image, return_tensors="pt")

        with torch.no_grad():
            logits = IMAGE_MODEL(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            nsfw_score = float(probs[0][nsfw_idx].item())

        predicted_idx = int(torch.argmax(probs, dim=-1).item())
        predicted_label = str(IMAGE_MODEL.config.id2label.get(predicted_idx, "unknown"))

        results.append(
            ModerationResult(
                filename=upload.filename or "unknown",
                label=predicted_label,
                nsfwScore=nsfw_score,
                isSensitive=nsfw_score >= threshold,
            )
        )

    flags = [
        f"sensitive_image_detected:{item.filename}:nsfw_score={item.nsfwScore:.4f}"
        for item in results
        if item.isSensitive
    ]

    return ModerationResponse(
        success=True,
        threshold=threshold,
        isSensitive=any(item.isSensitive for item in results),
        flags=flags,
        results=results,
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
