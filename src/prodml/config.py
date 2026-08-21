from functools import lru_cache
from pathlib import Path

from pydantic import Field
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

    year: int = Field(default=2024, ge=2009, le=2100)
    month: int = Field(default=1, ge=1, le=12)
    data_dir: Path = PROJECT_ROOT / "data"
    model_path: Path = PROJECT_ROOT / "models" / "model.pkl"
    report_path: Path = PROJECT_ROOT / "reports" / "module-1.md"
    validation_fraction: float = Field(default=0.20, gt=0, lt=1)
    data_url_override: str | None = Field(default=None, validation_alias="DATA_URL")

    @property
    def data_url(self) -> str:
        """Return the configured URL or the official monthly TLC URL."""
        if self.data_url_override:
            return self.data_url_override
        return (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"green_tripdata_{self.year}-{self.month:02d}.parquet"
        )

    @property
    def data_path(self) -> Path:
        return self.data_dir / f"green_tripdata_{self.year}-{self.month:02d}.parquet"

    @property
    def model_dir(self) -> Path:
        return self.model_path.parent

    @property
    def features(self) -> list[str]:
        return ["PU_DO", "trip_distance"]

    @property
    def target(self) -> str:
        return "duration"

    def ensure_project_directories(self) -> None:
        """Create directories required by the baseline workflow."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the process."""
    return Settings()
