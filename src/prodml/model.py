from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class modelBase(ABC):
    @abstractmethod
    def predict(self, X: list[float]) -> float:
        pass


class RideDurationNet(nn.Module):
    def __init__(self, in_features: int = 2, hidden: int = 16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RideDurationModel(modelBase):
    def __init__(self, model_path: str):
        self._model = RideDurationNet()

        state_dict = torch.load(
            model_path,
            weights_only=True
        )

        self._model.load_state_dict(state_dict)
        self._model.eval()

    def predict(self, X: list[float]) -> float:
        with torch.no_grad():
            tensor = torch.tensor(
                [X],
                dtype=torch.float32
            )

            return self._model(tensor).item()