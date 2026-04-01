import io
import os
from threading import Event, Lock, Thread
from typing import List

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
import logging

from recommend import select_best_recommender

logging.basicConfig(level=logging.INFO)


def _load_env() -> None:
    env_path = os.getenv("DOTENV_PATH") or os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)


_load_env()

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


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 20


class RecommendResponse(BaseModel):
    user_id: str
    count: int
    post_ids: List[str]


class RefreshRecommenderResponse(BaseModel):
    success: bool
    refreshed: bool
    error: str | None = None


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


RECOMMENDER = None
RECOMMENDER_INIT_ERROR = None
RECOMMENDER_LOCK = Lock()
RECOMMENDER_BUILD_LOCK = Lock()
RECOMMENDER_REFRESH_SECONDS = int(os.getenv("RECOMMENDER_REFRESH_SECONDS", "300"))
RECOMMENDER_STOP_EVENT = Event()
RECOMMENDER_REFRESH_THREAD = None


def _refresh_recommender_once() -> bool:
    global RECOMMENDER
    global RECOMMENDER_INIT_ERROR

    with RECOMMENDER_BUILD_LOCK:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            RECOMMENDER_INIT_ERROR = "Missing MONGO_URI environment variable"
            return False

        try:
            next_recommender, best_config, best_metrics = select_best_recommender(mongo_uri)
        except Exception as exc:
            RECOMMENDER_INIT_ERROR = str(exc)
            logging.exception("[recommender] Refresh failed")
            return False

        with RECOMMENDER_LOCK:
            RECOMMENDER = next_recommender
            RECOMMENDER_INIT_ERROR = None

        logging.info(
            "[recommender] Refresh succeeded with best model: factors=%s reg=%s iter=%s alpha=%s NDCG@%s=%.4f",
            best_config.factors,
            best_config.regularization,
            best_config.iterations,
            best_config.alpha,
            best_config.eval_k,
            best_metrics.get("ndcg_at_k", 0.0),
        )
        return True


def _recommender_refresh_loop() -> None:
    while not RECOMMENDER_STOP_EVENT.is_set():
        # Sleep first so startup request path can still initialize immediately.
        if RECOMMENDER_STOP_EVENT.wait(timeout=RECOMMENDER_REFRESH_SECONDS):
            break
        _refresh_recommender_once()


def _get_recommender():
    if RECOMMENDER is None and not _refresh_recommender_once():
        raise RuntimeError(RECOMMENDER_INIT_ERROR or "Failed to initialize recommender")
    return RECOMMENDER


@app.on_event("startup")
def startup_recommender_refresh() -> None:
    global RECOMMENDER_REFRESH_THREAD

    RECOMMENDER_STOP_EVENT.clear()
    _refresh_recommender_once()

    if RECOMMENDER_REFRESH_SECONDS <= 0:
        logging.info("[recommender] Auto refresh disabled (RECOMMENDER_REFRESH_SECONDS <= 0)")
        return

    RECOMMENDER_REFRESH_THREAD = Thread(target=_recommender_refresh_loop, daemon=True)
    RECOMMENDER_REFRESH_THREAD.start()
    logging.info(
        "[recommender] Auto refresh enabled every %s seconds",
        RECOMMENDER_REFRESH_SECONDS,
    )


@app.on_event("shutdown")
def shutdown_recommender_refresh() -> None:
    RECOMMENDER_STOP_EVENT.set()


@app.get("/health")
def health_check():
    return {
        "ok": IMAGE_MODEL is not None,
        "checkpointDir": CHECKPOINT_DIR,
        "error": MODEL_LOAD_ERROR,
        "recommendation": {
            "ready": RECOMMENDER is not None,
            "error": RECOMMENDER_INIT_ERROR,
        },
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend_posts(payload: RecommendRequest):
    if payload.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    try:
        recommender = _get_recommender()
    except RuntimeError as exc:
        logging.warning("[recommender] unavailable, returning empty recommendations: %s", exc)
        return RecommendResponse(
            user_id=str(payload.user_id),
            count=0,
            post_ids=[],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to initialize recommender: {exc}") from exc

    try:
        post_ids = recommender.recommend_post_ids(str(payload.user_id), k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to generate recommendations: {exc}") from exc

    return RecommendResponse(
        user_id=str(payload.user_id),
        count=len(post_ids),
        post_ids=post_ids,
    )


@app.post("/recommender/refresh", response_model=RefreshRecommenderResponse)
def refresh_recommender_now():
    refreshed = _refresh_recommender_once()
    return RefreshRecommenderResponse(
        success=refreshed,
        refreshed=refreshed,
        error=None if refreshed else RECOMMENDER_INIT_ERROR,
    )


@app.post("/moderate-images", response_model=ModerationResponse)
async def moderate_images(
    files: List[UploadFile] = File(...),
    threshold: float = Form(default=NSFW_THRESHOLD),
):
    logging.info(f"Received {len(files)} files for moderation.")
    if IMAGE_MODEL is None or IMAGE_PROCESSOR is None:
        logging.error(f"Model failed to load: {MODEL_LOAD_ERROR}")
        raise HTTPException(status_code=500, detail=f"Model failed to load: {MODEL_LOAD_ERROR}")

    if threshold < 0 or threshold > 1:
        logging.warning(f"Invalid threshold: {threshold}")
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    nsfw_idx = None
    for idx, label in IMAGE_MODEL.config.id2label.items():
        if str(label).strip().lower() == "nsfw":
            nsfw_idx = int(idx)
            break

    if nsfw_idx is None:
        logging.error("NSFW label is missing in model config")
        raise HTTPException(status_code=500, detail="NSFW label is missing in model config")

    results: List[ModerationResult] = []

    for upload in files:
        try:
            content = await upload.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            logging.error(f"Invalid image file: {upload.filename}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}")

        inputs = IMAGE_PROCESSOR(images=image, return_tensors="pt")

        with torch.no_grad():
            logits = IMAGE_MODEL(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            nsfw_score = float(probs[0][nsfw_idx].item())

        predicted_idx = int(torch.argmax(probs, dim=-1).item())
        predicted_label = str(IMAGE_MODEL.config.id2label.get(predicted_idx, "unknown"))

        logging.info(f"File: {upload.filename}, Predicted: {predicted_label}, NSFW Score: {nsfw_score}")

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

    logging.info(f"Sensitive images detected: {len(flags)}")

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
