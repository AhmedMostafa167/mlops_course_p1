from __future__ import annotations

import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from prodml.evaluate import Metrics
from prodml.features import get_target, to_feature_dicts
from prodml.model_types import ModelType


class ExperimentTracker:
    """Logs everything about a fitted model that MLflow's autologging doesn't already capture.

    Autologging (mlflow.sklearn.autolog / mlflow.xgboost.autolog) is expected to already be
    enabled by the caller for "lr" and "xgboost" runs — it handles their hyperparameters and
    the model artifact itself, plus XGBoost's own feature-importance plot for free. This class
    only logs what autolog can't: tags, validation-set metrics, the residual plot, the LR
    coefficient plot (autolog has no equivalent for plain sklearn linear models), the
    requirements snapshot, and — for "mlp" only, since autolog doesn't hook into raw PyTorch
    training loops — the params and model artifact autolog would otherwise have handled.
    """

    def __init__(
        self, model: Any, vectorizer: DictVectorizer, model_type: ModelType
    ) -> None:
        self.model = model
        self.vectorizer = vectorizer
        self.model_type = model_type

    def log_tags(self) -> None:
        framework = {"lr": "sklearn", "xgboost": "xgboost", "mlp": "pytorch"}[
            self.model_type
        ]
        mlflow.set_tags({"framework": framework, "author": "Ahmed Mostafa"})

    def log_params(self) -> None:
        """No-op for lr/xgboost — autolog already captures their hyperparameters."""
        if self.model_type != "mlp":
            return
        params = {
            "hidden_dim": self.model.network[0].out_features,
            "epochs": 250,
            "learning_rate": 1e-3,
        }
        mlflow.log_params({"model_type": self.model_type, **params})

    def log_metrics(self, metrics: Metrics) -> None:
        mlflow.log_metrics(metrics)

    def log_feature_importance_plot(self, n_top: int = 15) -> None:
        """LR only. XGBoost's autolog already logs its own feature_importance_weight.png;
        MLP has no native importance measure."""
        if self.model_type != "lr":
            return

        feature_names = self.vectorizer.get_feature_names_out()
        importances = np.abs(self.model.coef_)
        order = np.argsort(importances)[-n_top:]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feature_names[order], importances[order])
        ax.set_xlabel("|coefficient|")
        ax.set_title("Feature importance")
        fig.tight_layout()

        mlflow.log_figure(fig, "plots/feature_importance.png")
        plt.close(fig)

    def log_residual_plot(self, validation_df: Any) -> None:
        """Predicted vs. actual-minus-predicted — same for any model type."""
        X_valid = self.vectorizer.transform(to_feature_dicts(validation_df))
        y_valid = get_target(validation_df).to_numpy(dtype=np.float32)
        predictions = np.asarray(self.model.predict(X_valid)).reshape(-1)
        residuals = y_valid - predictions

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(predictions, residuals, alpha=0.3, s=10)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted duration")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title("Residual plot")
        fig.tight_layout()

        mlflow.log_figure(fig, "plots/residuals.png")
        plt.close(fig)

    def log_requirements_artifact(self) -> None:
        """Snapshot resolved dependency versions so this run's environment can be reproduced
        later, independent of what pyproject.toml resolves to by the time someone reruns it."""
        result = subprocess.run(
            ["uv", "export", "--no-hashes"],
            capture_output=True,
            text=True,
            check=True,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            req_path = Path(tmp_dir) / "requirements.txt"
            req_path.write_text(result.stdout)
            mlflow.log_artifact(str(req_path))

    def log_vectorizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vectorizer_path = Path(tmp_dir) / "vectorizer.pkl"
            with vectorizer_path.open("wb") as f_out:
                pickle.dump(self.vectorizer, f_out)
            mlflow.log_artifact(str(vectorizer_path), artifact_path="vectorizer")

    def log_model(self, input_example: np.ndarray | None = None) -> None:
        """No-op for lr/xgboost — autolog already logs their model artifact under 'model'.
        Manual only for mlp, since autolog doesn't hook into raw PyTorch training loops.
        Not registered to the Model Registry — that happens later, from tuning.py."""
        if self.model_type != "mlp":
            return
        if input_example is None:
            raise ValueError("input_example is required when logging an MLP")
        mlflow.pytorch.log_model(self.model, "model", input_example=input_example)

    def log_all(
        self,
        metrics: Metrics,
        validation_df: Any,
        input_example: np.ndarray | None = None,
    ) -> None:
        self.log_requirements_artifact()
        self.log_tags()
        self.log_params()
        self.log_metrics(metrics)
        self.log_residual_plot(validation_df)
        self.log_feature_importance_plot()
        self.log_vectorizer()
        self.log_model(input_example=input_example)
