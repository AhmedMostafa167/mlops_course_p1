from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

ModelType = Literal["lr", "xgboost", "mlp"]


class MLPRegressor(nn.Module):
    """Small dense neural network with an sklearn-like prediction method."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    @torch.no_grad()
    def predict(self, features: Any) -> Any:
        self.eval()
        tensor = torch.as_tensor(features, dtype=torch.float32)
        return self(tensor).squeeze(-1).cpu().numpy()


def validate_model_type(model_type: str) -> ModelType:
    if model_type not in {"lr", "xgboost", "mlp"}:
        raise ValueError(
            f"Unsupported model type {model_type!r}; choose lr, xgboost, or mlp"
        )
    return model_type  # type: ignore[return-value]


def infer_model_type(model: Any) -> ModelType:
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor

    if isinstance(model, LinearRegression):
        return "lr"
    if isinstance(model, XGBRegressor):
        return "xgboost"
    if isinstance(model, MLPRegressor):
        return "mlp"
    raise ValueError(f"Cannot infer model type for {type(model).__name__}")
