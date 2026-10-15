"""FastAPI service: image in, anomaly score + OK/KO decision out.

Run locally:
    uvicorn inspection.api:app --reload
"""

from __future__ import annotations

import io
import time
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inspection.config import MODEL_PATH, THRESHOLD_PATH
from inspection.threshold import load_threshold

app = FastAPI(title="Industrial anomaly inspection", version="0.1.0")

_model = None
_threshold: float | None = None


def _load() -> None:
    """Lazy-load model and threshold, so the app starts (and /health answers) without them."""
    global _model, _threshold
    if _model is None and MODEL_PATH.exists() and THRESHOLD_PATH.exists():
        from inspection.inference import AnomalyModel

        _model = AnomalyModel(MODEL_PATH)
        _threshold = load_threshold(THRESHOLD_PATH)


@app.get("/health")
def health() -> dict:
    _load()
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> dict:
    _load()
    if _model is None:
        raise HTTPException(status_code=503, detail="model not available")
    try:
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="not a valid image") from exc

    start = time.perf_counter()
    score, _anomaly_map = _model.predict(image)
    latency_ms = (time.perf_counter() - start) * 1000
    # TODO(week 3): return the anomaly map as a heatmap PNG (base64) for operators.
    return {
        "score": score,
        "threshold": _threshold,
        "decision": "KO" if score >= _threshold else "OK",
        "latency_ms": round(latency_ms, 1),
    }
