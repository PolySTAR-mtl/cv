from dataclasses import dataclass

import numpy as np

from polystar.common.models.box import Box
from polystar.common.models.object import Object
from polystar.common.target_pipeline.objects_validators.sequential_objects_validators_abc import (
    SequentialObjectsValidatorABC,
)


@dataclass
class InBoxValidator(SequentialObjectsValidatorABC):
    box: Box
    min_percentage_intersection: float

    def validate_single(self, obj: Object, image: np.ndarray) -> bool:
        aera = self.box.area_intersection(Box.from_size(obj.x, obj.y, obj.w, obj.h))
        threshold_aera_intersection = obj.w * obj.h * self.min_percentage_intersection
        return aera >= threshold_aera_intersection
