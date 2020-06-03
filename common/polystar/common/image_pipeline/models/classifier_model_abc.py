from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from polystar.common.image_pipeline.models.model_abc import ModelABC


@dataclass
class ClassifierModelABC(ModelABC, ABC):
    labels_: np.ndarray = field(init=False)
    label2index_: Dict[Any, int] = field(init=False, repr=False)

    def fit(self, features: List[Any], labels: List[Any]) -> "ClassifierModelABC":
        self.labels_ = np.asarray(sorted(set(labels)))
        self.label2index_ = {label: i for i, label in enumerate(self.labels_)}
        return self._fit_classifier(features, [self.label2index_[label] for label in labels], labels)

    @abstractmethod
    def predict_proba(self, features: List[Any]) -> np.ndarray:
        pass

    def predict(self, features: List[Any]) -> np.ndarray:
        indices = self.predict_proba(features).argmax(axis=1)
        return self.labels_[indices]

    def _fit_classifier(self, features: List[Any], label_indices: List[int], labels: List[Any]) -> "ClassifierModelABC":
        return self
