from abc import ABC, abstractmethod
from typing import Iterable, List

from polystar.common.models.image import Image
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot


class ObjectsLinkerABC(ABC):
    @abstractmethod
    def link_armors_to_robots(
        self, robots: List[DetectedRobot], armors: List[DetectedArmor], image: Image
    ) -> Iterable[DetectedRobot]:
        pass
