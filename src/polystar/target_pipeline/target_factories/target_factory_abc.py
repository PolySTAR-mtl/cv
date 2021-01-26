from abc import ABC, abstractmethod

import numpy as np

from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject
from polystar.target_pipeline.target_abc import TargetABC


class TargetFactoryABC(ABC):
    @abstractmethod
    def from_object(self, obj: DetectedROCOObject, image: np.array) -> TargetABC:
        pass
