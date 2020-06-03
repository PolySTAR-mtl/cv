from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class ModelABC(ABC):
    def fit(self, features: List[Any], labels: List[Any]) -> "ModelABC":
        return self

    @abstractmethod
    def predict(self, features: List[Any]) -> np.ndarray:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
