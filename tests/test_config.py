from prodml.config import PROJECT_ROOT, Settings


def test_model_artifacts_use_repository_models_directory():
    settings = Settings()

    assert settings.model_dir == PROJECT_ROOT / "models"
    assert settings.model_path == PROJECT_ROOT / "models" / "model.pkl"


def test_relative_path_overrides_are_resolved_from_repository_root():
    settings = Settings(
        MODEL_PATH="models/model.pkl",
        MODEL_DIR="models",
        DATA_DIR="data",
        REPORT_PATH="reports/module-1.md",
        EXPERIMENT_NAME="test-experiment",
        REGISTERED_MODEL_NAME="test-model",
        PRODUCTION_MODEL_URI="models:/test-model/Production",
    )

    assert settings.model_path == PROJECT_ROOT / "models" / "model.pkl"
    assert settings.model_dir == PROJECT_ROOT / "models"
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.report_path == PROJECT_ROOT / "reports" / "module-1.md"
