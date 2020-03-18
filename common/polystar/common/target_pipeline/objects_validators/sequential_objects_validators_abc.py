from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.common.models.object import Object
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC


class SequentialObjectsValidatorABC(ObjectsValidatorABC, ABC):
    def validate(self, objects: List[Object], image: np.ndarray) -> List[bool]:
        return [self.validate_single(obj, image) for obj in objects]

    @abstractmethod
    def validate_single(self, obj: Object, image: np.ndarray) -> bool:
        pass
