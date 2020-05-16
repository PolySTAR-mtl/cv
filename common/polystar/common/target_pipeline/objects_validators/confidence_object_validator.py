import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC


class ConfidenceObjectValidator(ObjectsValidatorABC[DetectedObject]):
    """Keep only objects for which we are confident enough."""

    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold

    def validate_single(self, obj: DetectedObject, image: np.ndarray) -> bool:
        return obj.confidence >= self.confidence_threshold
