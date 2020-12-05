from dataclasses import dataclass, field

import numpy as np
from numpy import argmax

from polystar.common.models.object import ArmorColor, ArmorDigit, ObjectType
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject


@dataclass
class DetectedArmor(DetectedObject):
    def __post_init__(self):
        assert self.type == ObjectType.Armor

    colors_proba: np.ndarray = field(init=False, default=None)
    digits_proba: np.ndarray = field(init=False, default=None)

    _color: ArmorColor = field(init=False, default=None)
    _digit: ArmorDigit = field(init=False, default=None)

    @property
    def color(self) -> ArmorColor:
        if self._color is not None:
            return self._color

        if self.colors_proba is not None:
            self._color = ArmorColor(self.colors_proba.argmax() + 1)
            return self._color

        return ArmorColor.Unknown

    @property
    def digit(self) -> ArmorDigit:
        if self._digit is not None:
            return self._digit

        if self.digits_proba:
            self._digit = ArmorDigit(argmax(self.colors_proba) + 1)
            return self._digit

        return ArmorDigit.UNKNOWN

    def __str__(self) -> str:
        return f"{self.type.name} ({self.confidence:.1%}; {self.color.short}; {self.digit.short})"
