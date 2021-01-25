from typing import Iterable

from numpy.core.multiarray import ndarray

from polystar.common.models.object import Armor
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC


class ArmorDigitValidator(ObjectsValidatorABC[DetectedRobot]):
    def __init__(self, digits: Iterable[int]):
        self.digits = digits

    def validate_single(self, armor: Armor, image: ndarray) -> bool:
        return isinstance(armor, Armor) and armor.number in self.digits
