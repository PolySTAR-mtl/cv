from dataclasses import dataclass, field
from typing import List

from polystar.models.roco_object import ObjectType
from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject


@dataclass
class DetectedRobot(DetectedROCOObject):
    armors: List[DetectedArmor] = field(default_factory=list)


class FakeDetectedRobot(DetectedRobot):
    def __init__(self, armor: DetectedArmor):
        super().__init__(type=ObjectType.CAR, box=armor.box, confidence=armor.confidence, armors=[armor])
