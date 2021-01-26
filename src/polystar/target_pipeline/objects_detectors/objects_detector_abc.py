from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.target_pipeline.detected_objects.detected_objects_factory import DetectedObjectFactory
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot


@dataclass
class ObjectsDetectorABC(ABC):
    objects_factory: DetectedObjectFactory

    @abstractmethod
    def detect(self, image: np.array) -> Tuple[List[DetectedRobot], List[DetectedArmor]]:
        pass
