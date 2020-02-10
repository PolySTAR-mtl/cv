import numpy as np

from polystar.common.models.object import ObjectType, Object
from polystar.common.pipeline.objects_validators.sequential_objects_validators_abc import SequentialObjectsValidatorABC


class TypeObjectValidator(SequentialObjectsValidatorABC):
    """Keep only the objects of a desired type"""

    def __init__(self, *desired_types: ObjectType):
        self.desired_types = set(desired_types)

    def validate_single(self, obj: Object, image: np.ndarray) -> bool:
        return obj.type in self.desired_types
