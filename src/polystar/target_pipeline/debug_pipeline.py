from dataclasses import dataclass, field
from typing import Iterable, List

from injector import inject

from polystar.models.image import Image
from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.target_pipeline.target_abc import TargetABC
from polystar.target_pipeline.target_pipeline import TargetPipeline, assert_any_armors_detected


@dataclass
class DebugInfo:
    image: Image
    detected_robots: List[DetectedRobot]
    validated_robots: List[DetectedRobot] = field(init=False, default_factory=list)
    selected_armor: DetectedArmor = field(init=False, default=None)
    target: TargetABC = field(init=False, default=None)


@inject
@dataclass
class DebugTargetPipeline(TargetPipeline):
    """Wrap a pipeline with debug, to store debug infos"""

    debug_info_: DebugInfo = field(init=False)

    def flow_debug(self, images: Iterable[Image]):
        for _ in self.flow_targets(images):
            yield self.debug_info_

    def _make_target_from_robots(self, image: Image, robots: List[DetectedRobot]) -> TargetABC:
        self.debug_info_ = DebugInfo(image, robots)
        self.debug_info_.validated_robots = self.robots_filters.filter(self.debug_info_.detected_robots)
        assert_any_armors_detected(self.debug_info_.validated_robots)
        self.debug_info_.selected_armor = self.object_selector.select(self.debug_info_.validated_robots, image)
        self.debug_info_.target = self.target_factory.from_object(self.debug_info_.selected_armor, image)
        return self.debug_info_.target
