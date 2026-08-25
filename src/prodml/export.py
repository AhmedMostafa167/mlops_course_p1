import numpy as np
import onnx
from onnxruntime import InferenceSession
from skl2onnx import to_onnx

from prodml.config import get_settings
from prodml.data import load_data, train_validation_split
from prodml.features import prepare_features, to_feature_dicts
from prodml.predict import DurationPredictor
import structlog
from prodml.logging_config import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)

settings = get_settings()
settings.ensure_project_directories()
logger.info("model_loading_started", path=str(settings.model_path))
predictor = DurationPredictor.load(settings.model_path)
logger.info("model_loaded", path=str(settings.model_path))
model = predictor.model
vectorizer = predictor.vectorizer

raw_data = load_data()
prepared_data = prepare_features(raw_data)
_, validation_df = train_validation_split(prepared_data)
validation_df = validation_df.head(500)
logger.info("validation_rows_selected", rows=len(validation_df))

feature_dicts = to_feature_dicts(validation_df)
X = vectorizer.transform(feature_dicts).toarray().astype(np.float32)
logger.info(
    "features_vectorized",
    rows=X.shape[0],
    columns=X.shape[1],
)

logger.info("onnx_export_started")
try:
    onnx_model = to_onnx(
        model,
        X[:1],
        target_opset=12,
    )
except Exception:
    logger.exception("onnx_export_failed")
    raise


logger.info("onnx_export_completed")

onnx.checker.check_model(onnx_model)
logger.info("onnx_model_validated")
onnx_path = settings.model_dir / "model.onnx"
onnx_path.write_bytes(onnx_model.SerializeToString())
logger.info("onnx_model_saved", path=str(onnx_path))

session = InferenceSession(
    str(onnx_path),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name

pred_skl = model.predict(X)
pred_ort = session.run(None, {input_name: X})[0]
pred_ort = np.asarray(pred_ort).reshape(-1)
pred_skl = np.asarray(pred_skl).reshape(-1)

max_difference = np.max(np.abs(pred_skl - pred_ort))

logger.info(
    "parity_check_completed",
    rows=len(pred_skl),
    max_difference=max_difference,
    tolerance=1e-4,
)
assert np.allclose(pred_skl, pred_ort, atol=1e-4)
