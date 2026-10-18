"""
AI Service Integration Prototype — FastAPI application.

Provides a REST API for property value estimation using a trained
scikit-learn model.  Training data is **synthetic**; predictions are
for demonstration purposes only.

Endpoints
---------
GET  /                 — Liveness probe (process is running).
GET  /api/v1/health    — Process health (always OK while alive).
GET  /api/v1/ready     — Readiness probe (model loaded & usable).
POST /api/v1/predict   — Property value estimation.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import sklearn
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_service")

API_KEY: str | None = os.environ.get("API_KEY") or None
RATE_LIMIT_MAX: int = int(os.environ.get("RATE_LIMIT_MAX", "120"))
RATE_LIMIT_WINDOW: int = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "").split(",")
    if o.strip()
]
MAX_REQUEST_SIZE: int = int(os.environ.get("MAX_REQUEST_SIZE", str(1024 * 16)))

_PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = _PROJECT_DIR / "model.joblib"
METADATA_PATH = _PROJECT_DIR / "model_metadata.json"


class PredictionRequest(BaseModel):
    """Input schema for property value estimation.

    All numeric fields reject NaN and Infinity.  Ranges are based on
    the synthetic training-data distribution (see *model_metadata.json*).

    Units
    -----
    - ``square_footage``: livable area in **square feet**.
    - ``age_years``: property age in **years**.
    - ``num_bathrooms``: accepts half-increments (1.0, 1.5, 2.0, …).
    """

    square_footage: float = Field(
        ..., gt=200, le=20_000,
        description="Livable area in square feet (200 < value ≤ 20,000).",
        examples=[1500.0],
    )
    num_bedrooms: int = Field(
        ..., ge=1, le=20,
        description="Number of bedrooms (1–20).",
        examples=[3],
    )
    num_bathrooms: float = Field(
        ..., ge=1.0, le=10.0,
        description="Number of bathrooms in 0.5 increments (1.0–10.0).",
        examples=[2.0],
    )
    age_years: int = Field(
        ..., ge=0, le=200,
        description="Age of the property in years (0–200).",
        examples=[10],
    )
    location_tier: Literal["urban", "suburban", "rural"] = Field(
        ...,
        description="Location classification.",
        examples=["suburban"],
    )
    has_garage: bool = Field(
        ...,
        description="Whether the property has a garage.",
        examples=[True],
    )

    @field_validator("square_footage", "num_bathrooms", mode="before")
    @classmethod
    def reject_non_finite(cls, v: Any) -> Any:
        """Reject NaN and Infinity before any other validation."""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Value must be a finite number.")
        return v

    @field_validator("num_bathrooms")
    @classmethod
    def validate_bathrooms_step(cls, v: float) -> float:
        if (v * 2) % 1 != 0:
            raise ValueError("Must be a multiple of 0.5.")
        return v

    model_config = {"extra": "forbid"}


class PredictionResponse(BaseModel):
    """Output schema for property value estimation."""

    status: str = "success"
    predicted_value_usd: float = Field(
        ..., description="Estimated property value in US dollars.",
    )
    currency: str = Field(
        default="USD", description="Currency of the predicted value.",
    )
    model_version: str = Field(
        ..., description="Version identifier of the model used.",
    )
    extrapolation_warning: bool = Field(
        default=False,
        description=(
            "True when any input falls outside the training-data range, "
            "meaning the prediction may be unreliable."
        ),
    )
    input_summary: dict = Field(
        ..., description="Echo of the validated input features.",
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    uptime_seconds: float | None = None


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    sklearn_version: str


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
    request_id: str | None = None


class RateLimiter:
    """Simple per-key sliding-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._requests: dict[str, collections.deque] = {}

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        if key not in self._requests:
            self._requests[key] = collections.deque()
        q = self._requests[key]
        while q and q[0] < now - self._window:
            q.popleft()
        if len(q) >= self._max:
            return False
        q.append(now)
        return True


class PropertyPredictor:
    """Wraps the trained sklearn Pipeline with metadata & safety checks.

    Security note
    -------------
    ``joblib.load`` can execute arbitrary code.  The model file path is
    hard-coded relative to this source file and must **never** be
    controlled by external user input.
    """

    _DEFAULT_RANGES: dict[str, dict] = {
        "square_footage": {"min": 300.0, "max": 10_000.0},
        "num_bedrooms": {"min": 1, "max": 9},
        "num_bathrooms": {"min": 1.0, "max": 4.0},
        "age_years": {"min": 0, "max": 79},
    }

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Run 'python train_model.py' first."
            )

        self._pipeline = joblib.load(model_path)
        self._loaded_at: float = time.time()
        self._model_version: str = "unknown"
        self._training_ranges: dict[str, dict] = dict(self._DEFAULT_RANGES)
        self._metadata: dict = {}

        # Load optional metadata
        metadata_path = model_path.parent / "model_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                self._model_version = self._metadata.get("model_version", "unknown")

                # Refresh training ranges from saved metadata
                saved_ranges = self._metadata.get("feature_ranges", {})
                for key in self._training_ranges:
                    if key in saved_ranges:
                        self._training_ranges[key] = saved_ranges[key]

                # Warn on sklearn version mismatch
                saved_sklearn = self._metadata.get("sklearn_version")
                if saved_sklearn and saved_sklearn != sklearn.__version__:
                    logger.warning(
                        "Model trained with sklearn %s; runtime has %s. "
                        "This may cause compatibility issues.",
                        saved_sklearn, sklearn.__version__,
                    )
            except Exception:
                logger.warning("Could not parse model metadata at %s", metadata_path)
        else:
            logger.warning("Model metadata not found at %s", metadata_path)

        logger.info(
            "Model loaded: version=%s, file=%s",
            self._model_version, model_path.name,
        )

    @property
    def version(self) -> str:
        return self._model_version

    @property
    def metadata(self) -> dict:
        return self._metadata

    def _check_extrapolation(self, req: PredictionRequest) -> bool:
        """Return True if any input is outside the training-data range."""
        checks = {
            "square_footage": req.square_footage,
            "num_bedrooms": req.num_bedrooms,
            "num_bathrooms": req.num_bathrooms,
            "age_years": req.age_years,
        }
        for field_name, value in checks.items():
            bounds = self._training_ranges.get(field_name, {})
            if "min" in bounds and value < bounds["min"]:
                return True
            if "max" in bounds and value > bounds["max"]:
                return True
        return False

    def _encode_features(self, req: PredictionRequest) -> np.ndarray:
        """Encode a PredictionRequest into the feature array expected
        by the pipeline (must match training column order)."""
        return np.array([[
            req.square_footage,
            req.num_bedrooms,
            req.num_bathrooms,
            req.age_years,
            req.location_tier,
            int(req.has_garage),
        ]], dtype=object)

    def predict(self, req: PredictionRequest) -> dict:
        features = self._encode_features(req)
        predicted_value = float(self._pipeline.predict(features)[0])
        predicted_value = max(predicted_value, 10_000.0)

        return {
            "predicted_value_usd": round(predicted_value, 2),
            "currency": "USD",
            "model_version": self.version,
            "extrapolation_warning": self._check_extrapolation(req),
        }


_predictor: PropertyPredictor | None = None
_start_time: float = 0.0
_rate_limiter: RateLimiter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup; release resources on shutdown.

    If the model file is missing or corrupt the service starts in
    **degraded** mode — ``/api/v1/health`` remains OK but
    ``/api/v1/ready`` returns 503 and ``/api/v1/predict`` is disabled.
    """
    global _predictor, _start_time, _rate_limiter  # noqa: PLW0603
    _start_time = time.time()
    _rate_limiter = RateLimiter(RATE_LIMIT_MAX, RATE_LIMIT_WINDOW)

    logger.info("Starting AI Service — loading model …")
    try:
        _predictor = PropertyPredictor()
        logger.info("Model ready  (version %s).", _predictor.version)
    except FileNotFoundError as exc:
        logger.error("Model not available: %s", exc)
        logger.error("Service starting in DEGRADED mode (predictions disabled).")
        _predictor = None
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        _predictor = None

    yield

    logger.info("Shutting down AI Service.")


app = FastAPI(
    title="AI Service Integration Prototype",
    description=(
        "REST API for property value estimation using a trained ML model.\n\n"
        "**Prototype Notice:** This service uses *synthetic* training data. "
        "Predictions are for demonstration purposes only and do not represent "
        "real property market values."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    """Inject request-ID, enforce rate limits & request size."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    if _rate_limiter is not None:
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "status": "error",
                    "detail": "Rate limit exceeded. Try again later.",
                    "request_id": request_id,
                },
            )

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "status": "error",
                "detail": "Request body too large.",
                "request_id": request_id,
            },
        )

    # Health endpoints are always accessible (for load-balancer probes).
    if API_KEY is not None and request.url.path not in (
        "/", "/api/v1/health", "/api/v1/ready", "/docs", "/redoc", "/openapi.json",
    ):
        provided_key = request.headers.get("X-API-Key")
        if provided_key != API_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "status": "error",
                    "detail": "Invalid or missing API key.",
                    "request_id": request_id,
                },
            )

    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    logger.info(
        "%s %s → %d (%.3fs) [%s]",
        request.method, request.url.path,
        response.status_code, elapsed, request_id[:8],
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    messages = [
        f"[{' -> '.join(str(p) for p in e.get('loc', []))}] "
        f"{e.get('msg', 'invalid value')}"
        for e in exc.errors()
    ]
    detail = "; ".join(messages)
    request_id = getattr(request.state, "request_id", None)
    logger.warning("Validation error [%s]: %s", request_id and request_id[:8], detail)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "detail": detail, "request_id": request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    logger.error(
        "HTTP %s [%s]: %s",
        exc.status_code, request_id and request_id[:8], exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail, "request_id": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception,
) -> JSONResponse:
    """Catch-all — never leak stack-traces or internal paths to clients."""
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "Unhandled error [%s]: %s", request_id and request_id[:8], type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "detail": "An internal error occurred.",
            "request_id": request_id,
        },
    )


def get_predictor() -> PropertyPredictor:
    """Dependency: returns the loaded predictor or raises 503."""
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is not ready for predictions.",
        )
    return _predictor


PredictorDep = Annotated[PropertyPredictor, Depends(get_predictor)]


@app.get("/", tags=["Health"], summary="Root / liveness probe")
async def root() -> HealthResponse:
    """Basic liveness check — confirms the process is running."""
    return HealthResponse(status="ok", service="AI Service Integration Prototype")


@app.get(
    "/api/v1/health",
    tags=["Health"],
    summary="Process health",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    """Process-level health.  Always returns 200 while the process is
    alive, regardless of model state.  Use ``/api/v1/ready`` to check
    whether the model is loaded and predictions can be served."""
    return HealthResponse(
        status="healthy",
        service="AI Service Integration Prototype",
        uptime_seconds=round(time.time() - _start_time, 2) if _start_time else None,
    )


@app.get(
    "/api/v1/ready",
    tags=["Health"],
    summary="Model readiness",
    responses={
        200: {"description": "Model loaded and ready"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def ready():
    """Readiness probe — returns 200 only when the model is loaded and
    the service can serve predictions."""
    if _predictor is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "model_loaded": False,
                "model_version": None,
                "sklearn_version": sklearn.__version__,
            },
        )
    return ReadyResponse(
        status="ready",
        model_loaded=True,
        model_version=_predictor.version,
        sklearn_version=sklearn.__version__,
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Predict property value",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized (API key required)"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal prediction error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(
    payload: PredictionRequest,
    predictor: PredictorDep,
) -> PredictionResponse:
    """Estimate property value based on input features.

    Returns the predicted value in USD along with the model version.
    If any input feature is outside the training-data range an
    ``extrapolation_warning`` flag is set to ``true``.
    """
    logger.info(
        "Prediction request — sqft=%.1f, tier=%s",
        payload.square_footage,
        payload.location_tier,
    )
    try:
        result = predictor.predict(payload)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error.",
        )

    return PredictionResponse(
        status="success",
        predicted_value_usd=result["predicted_value_usd"],
        currency=result["currency"],
        model_version=result["model_version"],
        extrapolation_warning=result["extrapolation_warning"],
        input_summary=payload.model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
