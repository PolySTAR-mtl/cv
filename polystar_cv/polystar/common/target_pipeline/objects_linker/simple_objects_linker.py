from typing import Iterable, List

from polystar.common.models.image import Image
from polystar.common.models.object import ObjectType
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot, FakeDetectedRobot
from polystar.common.target_pipeline.objects_linker.objects_linker_abs import ObjectsLinkerABC
from polystar.common.target_pipeline.objects_validators.contains_box_validator import ContainsBoxValidator
from polystar.common.target_pipeline.objects_validators.negation_validator import NegationValidator
from polystar.common.target_pipeline.objects_validators.type_object_validator import TypeObjectValidator


class SimpleObjectsLinker(ObjectsLinkerABC):
    def __init__(self, min_percentage_intersection: float):
        super().__init__()
        self.min_percentage_intersection = min_percentage_intersection
        self.robots_filter = NegationValidator(TypeObjectValidator(ObjectType.ARMOR))
        self.armors_filter = TypeObjectValidator(ObjectType.ARMOR)

    def link_armors_to_robots(
        self, robots: List[DetectedRobot], armors: List[DetectedArmor], image: Image
    ) -> Iterable[DetectedRobot]:

        for armor in armors:
            parents_filter = ContainsBoxValidator[DetectedRobot](armor.box, self.min_percentage_intersection)
            parents = parents_filter.filter(robots, image)
            if len(parents) != 1:
                yield FakeDetectedRobot(armor)
            else:
                robot: DetectedRobot = parents[0]
                robot.armors.append(armor)

        for robot in robots:
            yield robot
