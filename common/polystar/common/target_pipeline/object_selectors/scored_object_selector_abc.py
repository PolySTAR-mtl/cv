from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC


class ScoredObjectSelectorABC(ObjectSelectorABC, ABC):
    def select(self, objects: List[DetectedObject], image: np.array) -> DetectedObject:
        return max(objects, key=lambda obj: self.score(obj, image))

    @abstractmethod
    def score(self, obj: DetectedObject, image: np.ndarray) -> float:
        pass
