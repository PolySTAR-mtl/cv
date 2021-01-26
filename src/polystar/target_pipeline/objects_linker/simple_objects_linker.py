from typing import Iterable, List

from polystar.models.image import Image
from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot, FakeDetectedRobot
from polystar.target_pipeline.objects_filters.contains_box_filter import ContainsBoxObjectsFilter
from polystar.target_pipeline.objects_linker.objects_linker_abs import ObjectsLinkerABC


class SimpleObjectsLinker(ObjectsLinkerABC):
    def __init__(self, min_percentage_intersection: float):
        super().__init__()
        self.min_percentage_intersection = min_percentage_intersection

    def link_armors_to_robots(
        self, robots: List[DetectedRobot], armors: List[DetectedArmor], image: Image
    ) -> Iterable[DetectedRobot]:

        for armor in armors:
            parents = ContainsBoxObjectsFilter(armor.box, self.min_percentage_intersection).filter(robots)
            if len(parents) != 1:
                yield FakeDetectedRobot(armor)
            else:
                robot: DetectedRobot = parents[0]
                robot.armors.append(armor)

        for robot in robots:
            yield robot
