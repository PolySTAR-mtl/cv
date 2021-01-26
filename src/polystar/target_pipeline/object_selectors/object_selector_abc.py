from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot


class ObjectSelectorABC(ABC):
    @abstractmethod
    def select(self, objects: List[DetectedRobot], image: np.array) -> DetectedArmor:
        pass
