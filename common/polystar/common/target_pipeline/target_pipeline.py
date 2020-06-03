from dataclasses import dataclass
from typing import List

from polystar.common.communication.target_sender_abc import TargetSenderABC
from polystar.common.models.image import Image
from polystar.common.target_pipeline.armors_descriptors.armors_descriptor_abc import ArmorsDescriptorABC
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.common.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC
from polystar.common.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC
from polystar.common.target_pipeline.objects_linker.objects_linker_abs import ObjectsLinkerABC
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC
from polystar.common.target_pipeline.target_abc import TargetABC
from polystar.common.target_pipeline.target_factories.target_factory_abc import TargetFactoryABC


class NoTargetFoundException(Exception):
    pass


@dataclass
class TargetPipeline:

    objects_detector: ObjectsDetectorABC
    armors_descriptors: List[ArmorsDescriptorABC]
    objects_linker: ObjectsLinkerABC
    objects_validators: List[ObjectsValidatorABC[DetectedRobot]]
    object_selector: ObjectSelectorABC
    target_factory: TargetFactoryABC
    target_sender: TargetSenderABC

    def predict_target(self, image: Image) -> TargetABC:
        selected_object = self.predict_best_object(image)
        target = self.target_factory.from_object(selected_object, image)
        self.target_sender.send(target)
        return target

    def predict_best_object(self, image: Image) -> DetectedObject:
        objects = self._get_robots_of_interest(image)
        selected_object = self.object_selector.select(objects, image)
        return selected_object

    def _get_robots_of_interest(self, image: Image) -> List[DetectedRobot]:
        robots = self._detect_robots(image)
        robots = self._filter_robots(image, robots)

        if not any(robot.armors for robot in robots):
            raise NoTargetFoundException()

        return robots

    def _filter_robots(self, image: Image, robots: List[DetectedRobot]) -> List[DetectedRobot]:
        for robots_validator in self.objects_validators:
            robots = robots_validator.filter(robots, image)
        return robots

    def _detect_robots(self, image: Image) -> List[DetectedRobot]:
        robots, armors = self.objects_detector.detect(image)
        for armors_descriptor in self.armors_descriptors:
            armors_descriptor.describe_armors(image, armors)
        return list(self.objects_linker.link_armors_to_robots(robots, armors, image))
