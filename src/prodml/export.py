from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import structlog
from onnxruntime import InferenceSession
from skl2onnx import to_onnx

from prodml.config import get_settings
from prodml.data import load_data, train_validation_split
from prodml.features import prepare_features, to_feature_dicts
from prodml.logging_config import configure_logging
from prodml.predict import DurationPredictor

configure_logging()
logger = structlog.get_logger(__name__)


def export_and_verify(
    predictor: DurationPredictor, validation_df: Any, n_rows: int = 500
) -> float:
    """Export predictor.model to ONNX, validate it, and return the max pickle/ONNX prediction gap."""
    settings = get_settings()
    validation_df = validation_df.head(n_rows)
    logger.info("validation_rows_selected", rows=len(validation_df))

    feature_dicts = to_feature_dicts(validation_df)
    X = predictor.vectorizer.transform(feature_dicts).toarray().astype(np.float32)
    logger.info("features_vectorized", rows=X.shape[0], columns=X.shape[1])

    logger.info("onnx_export_started")
    try:
        onnx_model = to_onnx(predictor.model, X[:1], target_opset=12)
    except Exception:
        logger.exception("onnx_export_failed")
        raise
    logger.info("onnx_export_completed")

    onnx.checker.check_model(onnx_model)
    logger.info("onnx_model_validated")

    onnx_path = settings.model_dir / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())
    logger.info("onnx_model_saved", path=str(onnx_path))

    session = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    pred_skl = np.asarray(predictor.model.predict(X)).reshape(-1)
    pred_ort = np.asarray(session.run(None, {input_name: X})[0]).reshape(-1)
    max_difference = float(np.max(np.abs(pred_skl - pred_ort)))

    logger.info(
        "parity_check_completed",
        rows=len(pred_skl),
        max_difference=max_difference,
        tolerance=1e-4,
    )
    return max_difference


def main() -> None:
    settings = get_settings()
    settings.ensure_project_directories()

    logger.info("model_loading_started", path=str(settings.model_path))
    predictor = DurationPredictor.load(settings.model_path)
    logger.info("model_loaded", path=str(settings.model_path))

    raw_data = load_data()
    prepared_data = prepare_features(raw_data)
    _, validation_df = train_validation_split(prepared_data)

    max_difference = export_and_verify(predictor, validation_df, n_rows=500)
    assert max_difference < 1e-4, f"ONNX parity failed: max difference {max_difference}"
    logger.info("onnx_export_verified", max_difference=max_difference)


if __name__ == "__main__":
    main()
