from abc import ABC, abstractmethod
from typing import Generic, List, Sequence

from sklearn.base import BaseEstimator

from polystar.pipeline.pipe_abc import IT
from polystar.utils.named_mixin import NamedMixin


class ClassifierABC(BaseEstimator, NamedMixin, Generic[IT], ABC):
    n_classes: int

    def fit(self, examples: List[IT], label_indices: List[int], validation_size: int) -> "ClassifierABC":
        return self

    @abstractmethod
    def predict_proba(self, examples: List[IT]) -> Sequence[float]:
        pass
