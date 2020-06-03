from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from polystar.common.image_pipeline.models.classifier_model_abc import ClassifierModelABC


class AbsoluteClassifierModelABC(ClassifierModelABC, ABC):
    @abstractmethod
    def predict(self, features: List[Any]) -> np.ndarray:
        pass

    def predict_proba(self, features: List[Any]) -> np.ndarray:
        proba = np.zeros((len(features), len(self.labels_)))
        preds = self.predict(features)
        indices = [self.label2index_[pred] for pred in preds]
        proba[(range(len(features)), indices)] = 1
        return proba
