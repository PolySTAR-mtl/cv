from dataclasses import dataclass, field

import numpy as np

from polystar.models.roco_object import ArmorColor, ArmorDigit, ObjectType
from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject
from research.armors.armor_color.pipeline import ArmorColorPipeline
from research.armors.armor_digit.pipeline import ArmorDigitPipeline


@dataclass
class DetectedArmor(DetectedROCOObject):
    def __post_init__(self):
        assert self.type == ObjectType.ARMOR

    colors_proba: np.ndarray = field(init=False, default=None)
    digits_proba: np.ndarray = field(init=False, default=None)

    _color: ArmorColor = field(init=False, default=None)
    _digit: ArmorDigit = field(init=False, default=None)

    @property
    def color(self) -> ArmorColor:
        if self._color is not None:
            return self._color

        if self.colors_proba is not None:
            self._color = ArmorDigitPipeline.classes[self.colors_proba.argmax()]
            return self._color

        return ArmorColor.UNKNOWN

    @property
    def digit(self) -> ArmorDigit:
        if self._digit is not None:
            return self._digit

        if self.digits_proba is not None:
            self._digit = ArmorColorPipeline.classes[self.digits_proba.argmax()]
            return self._digit

        return ArmorDigit.UNKNOWN

    def __str__(self) -> str:
        return f"{self.type.name} ({self.confidence:.1%}; {self.color.short}; {self.digit.short})"
