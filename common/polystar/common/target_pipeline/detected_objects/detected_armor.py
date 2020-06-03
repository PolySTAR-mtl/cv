from dataclasses import dataclass, field

import numpy as np
from numpy import argmax

from polystar.common.models.object import ORDERED_ARMOR_COLORS, ArmorColor, ObjectType
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject


@dataclass
class DetectedArmor(DetectedObject):
    def __post_init__(self):
        assert self.type == ObjectType.Armor

    colors_proba: np.ndarray = field(init=False, default=None)
    numbers_proba: np.ndarray = field(init=False, default=None)

    _color: ArmorColor = field(init=False, default=None)
    _number: int = field(init=False, default=None)

    @property
    def color(self) -> ArmorColor:
        if self._color is not None:
            return self._color

        if self.colors_proba is not None:
            self._color = ORDERED_ARMOR_COLORS[self.colors_proba.argmax()]
            return self._color

        return ArmorColor.Unknown

    @property
    def number(self) -> int:
        if self._number is not None:
            return self._number

        if self.numbers_proba:
            # FIXME: We skip some of the numbers at training...
            self._number = 1 + argmax(self.colors_proba)
            return self._number

        return 0

    def __str__(self) -> str:
        return (
            f"{self.type.name} "
            f"("
            f"{self.confidence:.1%}; "
            f"{self.color.name[0] if self.color is not ArmorColor.Unknown else '?'}; "
            f"{self.number or '?'}"
            f")"
        )
