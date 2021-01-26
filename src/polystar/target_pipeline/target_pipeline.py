from dataclasses import dataclass
from typing import List

from injector import inject

from polystar.communication.target_sender_abc import TargetSenderABC
from polystar.models.image import Image
from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC
from polystar.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC
from polystar.target_pipeline.objects_linker.objects_linker_abs import ObjectsLinkerABC
from polystar.target_pipeline.target_abc import TargetABC
from polystar.target_pipeline.target_factories.target_factory_abc import TargetFactoryABC


class NoTargetFoundException(Exception):
    pass


@inject
@dataclass
class TargetPipeline:

    objects_detector: ObjectsDetectorABC
    objects_linker: ObjectsLinkerABC
    objects_filters: List[ObjectsFilterABC]
    object_selector: ObjectSelectorABC
    target_factory: TargetFactoryABC
    target_sender: TargetSenderABC

    def predict_target(self, image: Image) -> TargetABC:
        selected_object = self.predict_best_object(image)
        target = self.target_factory.from_object(selected_object, image)
        self.target_sender.send(target)
        return target

    def predict_best_object(self, image: Image) -> DetectedROCOObject:
        objects = self._get_robots_of_interest(image)
        selected_object = self.object_selector.select(objects, image)
        return selected_object

    def _get_robots_of_interest(self, image: Image) -> List[DetectedRobot]:
        robots = self._detect_robots(image)
        robots = self._filter_robots(robots)

        if not any(robot.armors for robot in robots):
            raise NoTargetFoundException()

        return robots

    def _filter_robots(self, robots: List[DetectedRobot]) -> List[DetectedRobot]:
        for robots_validator in self.objects_filters:
            robots = robots_validator.filter(robots)
        return robots

    def _detect_robots(self, image: Image) -> List[DetectedRobot]:
        robots, armors = self.objects_detector.detect(image)
        return list(self.objects_linker.link_armors_to_robots(robots, armors, image))
