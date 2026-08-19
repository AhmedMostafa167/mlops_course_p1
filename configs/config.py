from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    MODEL_PATH: str = "models/model_full.pt"

    @property
    def model_path_resolved(self) -> Path:
        return PROJECT_ROOT / self.MODEL_PATH

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()