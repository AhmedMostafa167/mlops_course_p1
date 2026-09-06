from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import structlog
import torch
from onnxruntime import InferenceSession
from skl2onnx import to_onnx

from prodml.config import get_settings
from prodml.data import load_data, train_validation_split
from prodml.features import prepare_features, to_feature_dicts
from prodml.logging_config import configure_logging
from prodml.model_types import ModelType, infer_model_type
from prodml.predict import DurationPredictor

configure_logging()
logger = structlog.get_logger(__name__)


def _register_xgboost_converter() -> None:
    """skl2onnx has no built-in XGBoost support; wire in onnxmltools' converter once."""
    from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import (
        calculate_linear_regressor_output_shapes,
    )
    from xgboost import XGBRegressor

    update_registered_converter(
        XGBRegressor,
        "XGBoostXGBRegressor",
        calculate_linear_regressor_output_shapes,
        convert_xgboost,
    )


def _export_sklearn_compatible(
    model: Any, x_sample: np.ndarray, onnx_path: Path
) -> None:
    """lr and xgboost both implement the sklearn estimator interface skl2onnx expects."""
    if type(model).__name__ == "XGBRegressor":
        _register_xgboost_converter()
        # onnxmltools' tree-ensemble ops need the ai.onnx.ml domain pinned explicitly —
        # this skl2onnx build only supports up to version 3 there.
        target_opset = {"": 12, "ai.onnx.ml": 3}
    else:
        target_opset = 12
    onnx_model = to_onnx(model, x_sample, target_opset=target_opset)
    onnx_path.write_bytes(onnx_model.SerializeToString())


def _export_mlp(model: torch.nn.Module, x_sample: np.ndarray, onnx_path: Path) -> None:
    dummy_input = torch.from_numpy(x_sample)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=12,
        input_names=["features"],
        output_names=["output"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )


_EXPORTERS = {
    "lr": _export_sklearn_compatible,
    "xgboost": _export_sklearn_compatible,
    "mlp": _export_mlp,
}


def export_and_verify(
    predictor: DurationPredictor, validation_df: Any, n_rows: int = 500
) -> float:
    """Export predictor.model (lr, xgboost, or mlp) to ONNX, validate it, and return
    the max pickle/ONNX prediction gap."""
    settings = get_settings()
    validation_df = validation_df.head(n_rows)
    logger.info("validation_rows_selected", rows=len(validation_df))

    feature_dicts = to_feature_dicts(validation_df)
    X = predictor.vectorizer.transform(feature_dicts).toarray().astype(np.float32)
    logger.info("features_vectorized", rows=X.shape[0], columns=X.shape[1])

    model_type: ModelType = infer_model_type(predictor.model)
    onnx_path = settings.model_dir / "model.onnx"

    logger.info("onnx_export_started", model_type=model_type)
    try:
        _EXPORTERS[model_type](predictor.model, X[:1], onnx_path)
    except Exception:
        logger.exception("onnx_export_failed", model_type=model_type)
        raise
    logger.info("onnx_export_completed", model_type=model_type, path=str(onnx_path))

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("onnx_model_validated")

    session = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    pred_native = np.asarray(predictor.model.predict(X)).reshape(-1)
    pred_ort = np.asarray(session.run(None, {input_name: X})[0]).reshape(-1)
    max_difference = float(np.max(np.abs(pred_native - pred_ort)))

    logger.info(
        "parity_check_completed",
        model_type=model_type,
        rows=len(pred_native),
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
