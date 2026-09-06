from __future__ import annotations

import time

import numpy as np
import structlog
from onnxruntime import InferenceSession

from prodml.config import get_settings
from prodml.data import load_data, train_validation_split
from prodml.features import prepare_features, to_feature_dicts
from prodml.logging_config import configure_logging
from prodml.predict import DurationPredictor

configure_logging()
logger = structlog.get_logger(__name__)


def _benchmark(fn, n_repeats: int = 50) -> tuple[float, float]:
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(np.percentile(arr, 95))


def main() -> None:
    settings = get_settings()
    predictor = DurationPredictor.load(settings.model_path)

    raw = load_data()
    prepared = prepare_features(raw)
    _, validation_df = train_validation_split(prepared)
    validation_df = validation_df.head(500)

    X_sparse = predictor.vectorizer.transform(to_feature_dicts(validation_df))

    # The Pickle/sklearn model can consume the sparse matrix directly.
    pkl_mean, pkl_p95 = _benchmark(lambda: predictor.model.predict(X_sparse))

    # ONNX Runtime requires a dense NumPy array, not a SciPy sparse matrix.
    X_dense = X_sparse.toarray().astype(np.float32, copy=False)

    session = InferenceSession(
        str(settings.model_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    onnx_mean, onnx_p95 = _benchmark(lambda: session.run(None, {input_name: X_dense}))

    session = InferenceSession(
        str(settings.model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    onnx_mean, onnx_p95 = _benchmark(lambda: session.run(None, {input_name: X_dense}))

    logger.info(
        "benchmark_completed",
        pickle=pkl_mean,
        pickle_p95=pkl_p95,
        onnx=onnx_mean,
        onnx_p95=onnx_p95,
    )


if __name__ == "__main__":
    main()
