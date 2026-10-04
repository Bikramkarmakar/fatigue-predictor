"""
app.py  –  FastAPI backend
--------------------------
Exposes:
  POST /predict          → returns predicted fatigue score
  GET  /health           → liveness check
  GET  /model-metrics    → returns training metrics
"""

import json
import os
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

# ── Load artefacts at startup ─────────────────────────────────────────────────
def _load_artefacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            "Model artefacts not found. Run `python train_model.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, scaler, metrics


MODEL, SCALER, METRICS = _load_artefacts()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fatigue Predictor API",
    description=(
        "Predicts next-day fatigue score (1–10) from bedtime screen usage, "
        "screen brightness, and late-day caffeine intake."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ─────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    bedtime_phone_minutes: float = Field(
        ..., ge=0, le=300, description="Minutes spent on phone at bedtime (0–300)"
    )
    screen_brightness_pct: float = Field(
        ..., ge=0, le=100, description="Screen brightness percentage (0–100)"
    )
    caffeine_post_5pm_mg: float = Field(
        ..., ge=0, le=400, description="Caffeine consumed after 5 PM in mg (0–400)"
    )


class PredictResponse(BaseModel):
    next_day_fatigue_score: float
    fatigue_label: str
    confidence_note: str


def _fatigue_label(score: float) -> str:
    if score <= 2.5:
        return "Very Low Fatigue 😊"
    elif score <= 4.5:
        return "Low Fatigue 🙂"
    elif score <= 6.5:
        return "Moderate Fatigue 😐"
    elif score <= 8.5:
        return "High Fatigue 😴"
    else:
        return "Severe Fatigue 🥴"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-metrics")
def model_metrics():
    return METRICS


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        features = np.array(
            [[req.bedtime_phone_minutes, req.screen_brightness_pct, req.caffeine_post_5pm_mg]]
        )
        scaled = SCALER.transform(features)
        raw_score = float(MODEL.predict(scaled)[0])
        # Clamp to the known scale [1, 10]
        score = round(max(1.0, min(10.0, raw_score)), 2)
        return PredictResponse(
            next_day_fatigue_score=score,
            fatigue_label=_fatigue_label(score),
            confidence_note=f"Model R² = {METRICS.get('r2', 'N/A')}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
