from dataclasses import dataclass, field
from typing import List

from polystar.common.models.image import Image
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.common.target_pipeline.target_abc import TargetABC
from polystar.common.target_pipeline.target_pipeline import TargetPipeline


@dataclass
class DebugInfo:
    image: Image = None
    detected_robots: List[DetectedRobot] = field(init=False, default_factory=list)
    validated_robots: List[DetectedRobot] = field(init=False, default_factory=list)
    selected_armor: DetectedArmor = field(init=False, default=None)
    target: TargetABC = field(init=False, default=None)


@dataclass
class DebugTargetPipeline(TargetPipeline):
    """Wrap a pipeline with debug, to store debug infos"""

    debug_info_: DebugInfo = field(init=False, default_factory=DebugInfo)

    def predict_target(self, image: Image) -> TargetABC:
        self.debug_info_ = DebugInfo(image)
        target = super().predict_target(image)
        self.debug_info_.target = target
        return target

    def predict_best_object(self, image: Image) -> DetectedObject:
        best_object = super().predict_best_object(image)
        self.debug_info_.selected_armor = best_object
        return best_object

    def _get_robots_of_interest(self, image: Image) -> List[DetectedRobot]:
        objects = super()._get_robots_of_interest(image)
        self.debug_info_.validated_robots = objects
        return objects

    def _detect_robots(self, image) -> List[DetectedRobot]:
        objects = super()._detect_robots(image)
        self.debug_info_.detected_robots = objects
        return objects
