import numpy as np
from dataclasses import dataclass

from polystar.common.models.box import Box
from polystar.common.models.object import Object
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC


@dataclass
class InBoxValidator(ObjectsValidatorABC[Object]):
    box: Box
    min_percentage_intersection: float

    def validate_single(self, obj: Object, image: np.ndarray) -> bool:
        aera = self.box.area_intersection(obj.box)
        threshold_aera_intersection = obj.box.area * self.min_percentage_intersection
        return aera >= threshold_aera_intersection
