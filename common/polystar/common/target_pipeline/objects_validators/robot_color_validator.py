from dataclasses import dataclass

import numpy as np

from polystar.common.models.object import ArmorColor
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC


@dataclass
class RobotPercentageColorValidator(ObjectsValidatorABC[DetectedRobot]):
    color: ArmorColor
    min_percentage: 0.5

    def validate_single(self, robot: DetectedRobot, image: np.ndarray) -> bool:
        good_colors = [armor.color is self.color for armor in robot.armors]
        return sum(good_colors) >= len(good_colors) * self.min_percentage
