from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence, List

import numpy as np

from polystar.common.image_pipeline.models.model_abc import ModelABC


@dataclass
class RandomModel(ModelABC):
    labels_: np.ndarray = field(init=False, default=None)
    weights_: np.ndarray = field(init=False, default=None)

    def fit(self, features: Any, labels: List[Any]) -> "RandomModel":
        counter = Counter(labels)
        self.labels_ = np.asarray(list(counter.keys()))
        occurrences = np.asarray(list(counter.values()))
        self.weights_ = occurrences / occurrences.sum()
        return self

    def predict(self, features: Any) -> Sequence[Any]:
        return np.random.choice(self.labels_, size=len(features), replace=True, p=self.weights_)

    def __str__(self) -> str:
        return "random"
