from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.common.models.object import Object
from polystar.common.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC


class ScoredObjectSelectorABC(ObjectSelectorABC, ABC):
    def select(self, objects: List[Object], image: np.array) -> Object:
        return max(objects, key=lambda obj: self.score(obj, image))

    @abstractmethod
    def score(self, obj: Object, image: np.ndarray) -> float:
        pass
