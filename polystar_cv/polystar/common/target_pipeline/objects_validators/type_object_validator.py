import numpy as np

from polystar.common.models.object import ObjectType, Object
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC


class TypeObjectValidator(ObjectsValidatorABC[Object]):
    """Keep only the objects of a desired type"""

    def __init__(self, *desired_types: ObjectType):
        self.desired_types = set(desired_types)

    def validate_single(self, obj: Object, image: np.ndarray) -> bool:
        return obj.type in self.desired_types
