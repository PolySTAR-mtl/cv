import numpy as np

from polystar.common.models.object import Object
from polystar.common.target_pipeline.objects_validators.sequential_objects_validators_abc import (
    SequentialObjectsValidatorABC,
)


class ConfidenceObjectValidator(SequentialObjectsValidatorABC):
    """Keep only objects for which we are confident enough."""

    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold

    def validate_single(self, obj: Object, image: np.ndarray) -> bool:
        return obj.confidence >= self.confidence_threshold
