import mlflow
from mlflow.exceptions import MlflowException

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import optuna
import numpy as np
from prodml.data import load_data, train_validation_split
from prodml.features import get_target, prepare_features, to_feature_dicts
from prodml.logging_config import configure_logging
from prodml.train import fit_xgboost
from prodml.evaluate import evaluate_model
from prodml.config import get_settings
from structlog import get_logger
from prodml.logging_config import configure_logging

configure_logging()
logger = get_logger(__name__)


plain_data = load_data()
prepared = prepare_features(plain_data)
train_df, validation_df = train_validation_split(prepared)
vectorizer = DictVectorizer(sparse=True, dtype=np.float32)

X_train = vectorizer.fit_transform(to_feature_dicts(train_df))
y_train = get_target(train_df).to_numpy(dtype=np.float32)

X_val = vectorizer.transform(to_feature_dicts(validation_df))
y_val = get_target(validation_df).to_numpy(dtype=np.float32)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    with mlflow.start_run(nested=True):
        model = fit_xgboost(X_train, y_train, **params)
        metrics = evaluate_model(model, vectorizer, validation_df)
        mlflow.log_metrics(metrics)
        return metrics["mae"]


def main():
    settings = get_settings()
    settings.export_s3_env()
    logger.info(
        "mlflow_tracking_setup_started",
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    )
    mlflow.set_tracking_uri(uri=settings.MLFLOW_TRACKING_URI)
    logger.info(
        "mlflow_experiment_setup_started",
        experiment_name=settings.EXPERIMENT_NAME,
    )
    mlflow.set_experiment(settings.EXPERIMENT_NAME)
    logger.info("mlflow_experiment_setup_completed")

    mlflow.xgboost.autolog(log_models=False, log_datasets=False)

    with mlflow.start_run() as run:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)

        logger.info("best_params", **study.best_params)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_score", study.best_value)
        model = fit_xgboost(X_train, y_train, **study.best_params)
        model.fit(X_train, y_train)
        final_metrics = evaluate_model(model, vectorizer, validation_df)
        mlflow.log_metrics(final_metrics)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            model_format="json",
        )

        candidate_run_id = run.info.run_id

    model_version = mlflow.register_model(
        model_uri=f"runs:/{candidate_run_id}/model",
        name=settings.REGISTERED_MODEL_NAME,
    )
    logger.info("model_registered", model_version=model_version)


if __name__ == "__main__":
    main()
