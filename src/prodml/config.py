from functools import lru_cache
from pathlib import Path
import os
from functools import lru_cache
from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-backed settings for the Module 1 baseline workflow."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    year: int = Field(default=2024)
    month: int = Field(default=1, ge=1, le=12)
    features: list[str] = Field(default_factory=lambda: ["PU_DO", "trip_distance"])
    target: str = Field(default="duration")
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"
    model_path: Path = PROJECT_ROOT / "models" / "model.pkl"
    report_path: Path = PROJECT_ROOT / "reports" / "module-1.md"
    validation_fraction: float = Field(default=0.20, gt=0, lt=1)
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        validation_alias=AliasChoices("MLFLOW_TRACKING_URI", "TRACKING_URI"),
    )
    EXPERIMENT_NAME: str
    REGISTERED_MODEL_NAME: str
    PRODUCTION_MODEL_URI: str
    MLFLOW_S3_ENDPOINT_URL: str = Field(default="http://localhost:9000")
    AWS_ACCESS_KEY_ID: str = Field(default="minio_user")
    AWS_SECRET_ACCESS_KEY: str = Field(default="minio_password")
    @property
    def TRACKING_URI(self) -> str:
        """Backward-compatible alias for the MLflow tracking URI."""
        return self.MLFLOW_TRACKING_URI
    @property
    def data_url(self) -> str:
        """Return the configured URL."""
        return (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"green_tripdata_{self.year}-{self.month:02d}.parquet"
        )

    @property
    def data_path(self) -> Path:
        return self.data_dir / f"green_tripdata_{self.year}-{self.month:02d}.parquet"

    def ensure_project_directories(self) -> None:
        """Create directories required by the baseline workflow."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def export_s3_env(self) -> None:
        """Push S3 credentials into the process environment — boto3/MLflow's S3 client
        reads os.environ directly, it doesn't go through this Settings object."""
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", self.MLFLOW_S3_ENDPOINT_URL)
        os.environ.setdefault("AWS_ACCESS_KEY_ID", self.AWS_ACCESS_KEY_ID)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", self.AWS_SECRET_ACCESS_KEY)

@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the process."""
    return Settings()
