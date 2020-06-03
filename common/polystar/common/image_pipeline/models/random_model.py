from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Sequence

import numpy as np

from polystar.common.image_pipeline.models.absolute_classifier_model_abc import AbsoluteClassifierModelABC


@dataclass
class RandomModel(AbsoluteClassifierModelABC):
    labels_: np.ndarray = field(init=False, default=None)
    weights_: np.ndarray = field(init=False, default=None)

    def _fit_classifier(self, features: List[Any], label_indices: List[int], labels: List[Any]) -> "RandomModel":
        label2count = Counter(labels)
        occurrences = np.asarray([label2count[label] for label in self.labels_])
        self.weights_ = occurrences / occurrences.sum()
        return self

    def predict(self, features: List[Any]) -> Sequence[Any]:
        return np.random.choice(self.labels_, size=len(features), replace=True, p=self.weights_)

    def __str__(self) -> str:
        return "random"
