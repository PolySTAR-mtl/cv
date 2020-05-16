from typing import List

from dataclasses import dataclass, field

from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject


@dataclass
class DetectedRobot(DetectedObject):
    armors: List[DetectedArmor] = field(init=False, default_factory=list)
