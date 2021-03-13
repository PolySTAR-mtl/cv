from dataclasses import dataclass
from typing import Iterable, List, Optional

from injector import inject

from polystar.filters.filter_abc import FilterABC
from polystar.models.image import Image
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC
from polystar.target_pipeline.objects_detectors.robots_detector import RobotsDetector
from polystar.target_pipeline.target_abc import TargetABC
from polystar.target_pipeline.target_factories.target_factory_abc import TargetFactoryABC


class NoTargetFoundException(Exception):
    pass


@inject
@dataclass
class TargetPipeline:
    robots_detector: RobotsDetector
    robots_filters: FilterABC[DetectedRobot]
    object_selector: ObjectSelectorABC
    target_factory: TargetFactoryABC

    def flow_targets(self, images: Iterable[Image]) -> Iterable[Optional[TargetABC]]:
        for image, robots in self.robots_detector.flow_robots(images):
            try:
                yield self._make_target_from_robots(image, robots)
            except NoTargetFoundException:
                yield None

    def _make_target_from_robots(self, image: Image, robots: List[DetectedRobot]) -> TargetABC:
        robots = self.robots_filters.filter(robots)
        assert_any_armors_detected(robots)
        selected_armor = self.object_selector.select(robots, image)
        target = self.target_factory.from_object(selected_armor, image)
        return target


def assert_any_armors_detected(robots):
    if not any(robot.armors for robot in robots):
        raise NoTargetFoundException()
