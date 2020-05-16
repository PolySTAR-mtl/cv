from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject


class ObjectSelectorABC(ABC):
    @abstractmethod
    def select(self, objects: List[DetectedObject], image: np.array) -> DetectedObject:
        pass
