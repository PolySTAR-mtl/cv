from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject


class ObjectsDetectorABC(ABC):
    @abstractmethod
    def detect(self, image: np.array) -> List[DetectedObject]:
        pass
