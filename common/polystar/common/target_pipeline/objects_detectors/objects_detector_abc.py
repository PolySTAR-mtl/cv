from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot


class ObjectsDetectorABC(ABC):
    @abstractmethod
    def detect(self, image: np.array) -> Tuple[List[DetectedRobot], List[DetectedArmor]]:
        pass
