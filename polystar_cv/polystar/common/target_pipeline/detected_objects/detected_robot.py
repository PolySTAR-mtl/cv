from dataclasses import dataclass, field
from typing import List

from polystar.common.models.object import ObjectType
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject


@dataclass
class DetectedRobot(DetectedObject):
    armors: List[DetectedArmor] = field(default_factory=list)


class FakeDetectedRobot(DetectedRobot):
    def __init__(self, armor: DetectedArmor):
        super().__init__(type=ObjectType.CAR, box=armor.box, confidence=0, armors=[armor])
