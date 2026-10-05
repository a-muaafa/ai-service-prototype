from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_service")

class PredictionRequest(BaseModel):
    square_footage: float = Field(..., gt=200, le=20_000, examples=[1500.0])
    num_bedrooms: int = Field(..., ge=1, le=20, examples=[3])
    num_bathrooms: float = Field(..., ge=1.0, le=10.0, examples=[2.0])
    age_years: int = Field(..., ge=0, le=200, examples=[10])
    location_tier: Literal["urban", "suburban", "rural"] = Field(..., examples=["suburban"])
    has_garage: bool = Field(..., examples=[True])

    @field_validator("num_bathrooms")
    @classmethod
    def validate_bathrooms_step(cls, v: float) -> float:
        if (v * 2) % 1 != 0:
            raise ValueError("Must be a multiple of 0.5.")
        return v

class PredictionResponse(BaseModel):
    status: str
    predicted_value_usd: float
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    input_summary: dict
    model_version: str

class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str

MODEL_PATH = Path(__file__).parent / "model.joblib"
MODEL_VERSION = "2.0.0-gbr"

_MIN_CONFIDENCE: float = 55.0
_MAX_CONFIDENCE: float = 97.0

class PropertyPredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Run 'python train_model.py' first."
            )
        self._pipeline = joblib.load(model_path)
        self._loaded_at: float = time.time()
        logger.info("Model loaded from %s", model_path)

    @property
    def version(self) -> str:
        return MODEL_VERSION

    def _encode_features(self, request: PredictionRequest) -> np.ndarray:
        return np.array([[
            request.square_footage,
            request.num_bedrooms,
            request.num_bathrooms,
            request.age_years,
            request.location_tier,
            int(request.has_garage),
        ]], dtype=object)

    def _compute_confidence(self, square_footage: float, age_years: int) -> float:
        sqft_score = min(square_footage / 5_000.0, 1.0)
        age_score = 1.0 - min(age_years / 100.0, 1.0) * 0.4
        raw = sqft_score * 0.6 + age_score * 0.4
        return round(_MIN_CONFIDENCE + raw * (_MAX_CONFIDENCE - _MIN_CONFIDENCE), 2)

    def predict(self, request: PredictionRequest) -> dict:
        features = self._encode_features(request)
        predicted_value = float(self._pipeline.predict(features)[0])
        predicted_value = max(predicted_value, 10_000.0)

        return {
            "predicted_value_usd": round(predicted_value, 2),
            "confidence_score": self._compute_confidence(
                request.square_footage, request.age_years
            ),
            "model_version": self.version,
        }

_predictor: PropertyPredictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    logger.info("Starting AI Service - loading model.")
    _predictor = PropertyPredictor()
    logger.info("Model ready (version %s).", _predictor.version)
    yield
    logger.info("Shutting down AI Service.")

app = FastAPI(
    title="AI Service Integration Prototype",
    description="FastAPI service for property value estimation using a trained ML model.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    messages = [
        f"[{' -> '.join(str(p) for p in e.get('loc', []))}] {e.get('msg', 'invalid value')}"
        for e in exc.errors()
    ]
    detail = "; ".join(messages)
    logger.warning("Validation error on %s: %s", request.url.path, detail)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "detail": detail},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    logger.error("HTTP %s on %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail},
    )

def get_predictor() -> PropertyPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialized.",
        )
    return _predictor

PredictorDep = Annotated[PropertyPredictor, Depends(get_predictor)]

@app.get("/", tags=["Health"])
async def root() -> dict:
    return {"status": "ok", "service": "AI Service Integration Prototype"}

@app.get("/api/v1/health", tags=["Health"])
async def health(predictor: PredictorDep) -> dict:
    return {
        "status": "healthy",
        "model_version": predictor.version,
        "timestamp": time.time(),
    }

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def predict(
    payload: PredictionRequest,
    predictor: PredictorDep,
    request: Request,
) -> PredictionResponse:
    logger.info(
        "POST /api/v1/predict - sqft=%.1f tier=%s",
        payload.square_footage,
        payload.location_tier,
    )
    try:
        result = predictor.predict(payload)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error.",
        ) from exc

    return PredictionResponse(
        status="success",
        predicted_value_usd=result["predicted_value_usd"],
        confidence_score=result["confidence_score"],
        input_summary=payload.model_dump(),
        model_version=result["model_version"],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
