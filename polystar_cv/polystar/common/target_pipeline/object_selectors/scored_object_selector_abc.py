from abc import ABC, abstractmethod
from itertools import chain
from typing import List

import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.common.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC


class ScoredObjectSelectorABC(ObjectSelectorABC, ABC):
    def select(self, robots: List[DetectedRobot], image: np.array) -> DetectedArmor:
        return max(chain(*[robot.armors for robot in robots]), key=lambda obj: self.score(obj, image))

    @abstractmethod
    def score(self, armor: DetectedArmor, image: np.ndarray) -> float:
        pass
