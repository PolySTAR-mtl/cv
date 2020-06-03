from abc import ABC
from typing import List

from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor


class ArmorsDescriptorABC(ABC):
    def describe_armors(self, armors: List[DetectedArmor]):
        pass
