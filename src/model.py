from abc import ABC, abstractmethod
import torch

class modelBase(ABC):
    @abstractmethod
    def predict(self, X: list[float]) -> float:
        pass
class RideDurationModel(modelBase):
    def __init__(self, model_path: str = "model_full.pt"):
        self._model = torch.load(model_path, weights_only=False)
        self._model.eval()

    def predict(self, X: list[float]) -> float:
        with torch.no_grad():
            tensor = torch.tensor([X], dtype=torch.float32)
            return self._model(tensor).item()