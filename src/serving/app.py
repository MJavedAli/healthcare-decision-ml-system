from __future__ import annotations

from datetime import datetime
import time
from typing import Any
import logging

logger = logging.getLogger("mlops")
logging.basicConfig(level=logging.INFO)

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.common.schema import AppointmentFeatures, PredictionResponse
from src.monitoring.metrics import (
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_REQUESTS_TOTAL,
)
from src.serving.model_loader import load_metadata, load_model

app = FastAPI(title="No-Show Risk API", version="1.0.0")
model = None
metadata: dict[str, Any] = {}


@app.on_event("startup")
def startup() -> None:
    global model, metadata
    model = load_model()
    metadata = load_metadata()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": metadata.get("model_version", "unknown")}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: AppointmentFeatures) -> PredictionResponse:
    PREDICTION_REQUESTS_TOTAL.inc()
    start = time.time()
    try:
        input_df = pd.DataFrame(
            [
                {
                    "age": payload.age,
                    "prior_no_shows": payload.prior_no_shows,
                    "lead_time_hours": payload.lead_time_hours,
                    "num_previous_appointments": payload.num_previous_appointments,
                    "is_weekend": int(payload.is_weekend),
                    "distance_km": payload.distance_km,
                }
            ]
        )
        probability = float(model.predict_proba(input_df)[0][1])
        label = int(probability >= 0.5)
        logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "event": "prediction",
        "input": payload.dict(),
        "probability": probability,
        "label": label
        }))
        return PredictionResponse(
            no_show_probability=probability,
            predicted_label=label,
            model_version=str(metadata.get("model_version", "v1")),
        )
    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.inc()
        logger.error(f"Prediction failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PREDICTION_LATENCY_SECONDS.observe(time.time() - start)
