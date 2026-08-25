from prodml.config import PROJECT_ROOT, Settings


def test_model_artifacts_use_repository_models_directory():
    settings = Settings()

    assert settings.model_dir == PROJECT_ROOT / "models"
    assert settings.model_path == PROJECT_ROOT / "models" / "model.pkl"
