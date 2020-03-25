from abc import ABC, abstractmethod
from typing import Any, List, Sequence


class ModelABC(ABC):
    def fit(self, features: List[Any], labels: List[Any]) -> "ModelABC":
        return self

    @abstractmethod
    def predict(self, features: List[Any]) -> Sequence[Any]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
