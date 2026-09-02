from contextlib import asynccontextmanager
from datetime import datetime
from hashlib import sha256
from time import perf_counter
from typing import AsyncIterator
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from prodml.config import get_settings
from prodml.logging_config import configure_logging
from prodml.predict import DurationPredictor

from .schemas import (
    MetadataResponse,
    PredictionBatchRequest,
    PredictionBatchResponse,
    PredictionRequest,
    PredictionResponse,
)


configure_logging()
logger = structlog.get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach one correlation ID to every request and response."""

    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.correlation_id = correlation_id
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = correlation_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the predictor once when the application starts."""
    settings = get_settings()
    logger.info("model_loading_started", path=str(settings.model_path))
    app.state.predictor = DurationPredictor.load(settings.model_path)
    logger.info("model_loaded", path=str(settings.model_path))
    yield
    app.state.predictor = None


app = FastAPI(
    title="NYC Green Taxi Duration Service",
    lifespan=lifespan,
)
app.add_middleware(CorrelationIdMiddleware)


def get_predictor(request: Request) -> DurationPredictor:
    """Return the predictor loaded during application startup."""
    return request.app.state.predictor


def get_correlation_id(request: Request) -> str:
    """Return the correlation ID assigned to the current request."""
    return request.state.correlation_id


def get_model_metadata() -> MetadataResponse:
    """Build metadata for the currently configured model artifact."""
    settings = get_settings()
    artifact_bytes = settings.model_path.read_bytes()
    return MetadataResponse(
        model_version=f"linear-regression-{settings.year}-{settings.month:02d}",
        training_date=datetime.fromtimestamp(settings.model_path.stat().st_mtime)
        .date()
        .isoformat(),
        features=settings.features,
        framework="scikit-learn",
        artifact_hash=sha256(artifact_bytes).hexdigest(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a clean 422 response for invalid request data."""
    correlation_id = get_correlation_id(request)
    logger.warning(
        "request_validation_failed",
        path=request.url.path,
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Log unexpected errors and return a safe 500 response."""
    correlation_id = get_correlation_id(request)
    logger.exception(
        "unexpected_request_error",
        path=request.url.path,
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "correlation_id": correlation_id,
        },
    )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Return 200 only when the predictor is loaded in memory."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy"})
    return {"status": "healthy"}


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Return model version, training date, features, framework, and hash."""
    return get_model_metadata()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    predictor: DurationPredictor = Depends(get_predictor),
    correlation_id: str = Depends(get_correlation_id),
) -> PredictionResponse:
    """Return one taxi-duration prediction."""
    started = perf_counter()
    prediction = predictor.predict_one(
        {
            "PU_DO": payload.pu_do,
            "trip_distance": payload.trip_distance,
        }
    )
    latency_ms = (perf_counter() - started) * 1000
    settings = get_settings()
    return PredictionResponse(
        prediction=prediction,
        model_version=f"linear-regression-{settings.year}-{settings.month:02d}",
        correlation_id=correlation_id,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=PredictionBatchResponse)
def predict_batch(
    payload: PredictionBatchRequest,
    predictor: DurationPredictor = Depends(get_predictor),
    correlation_id: str = Depends(get_correlation_id),
) -> PredictionBatchResponse:
    """Return predictions for a list of taxi trips."""
    started = perf_counter()
    features = [
        {"PU_DO": item.pu_do, "trip_distance": item.trip_distance}
        for item in payload.items
    ]
    predictions = predictor.predict_batch(features)
    latency_ms = (perf_counter() - started) * 1000
    settings = get_settings()
    return PredictionBatchResponse(
        predictions=predictions,
        model_version=f"linear-regression-{settings.year}-{settings.month:02d}",
        correlation_id=correlation_id,
        latency_ms=round(latency_ms, 3),
    )
