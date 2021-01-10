from abc import ABC, abstractmethod

import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.target_abc import TargetABC


class TargetFactoryABC(ABC):
    @abstractmethod
    def from_object(self, obj: DetectedObject, image: np.array) -> TargetABC:
        pass
