from polystar.common.filters.filter_abc import FilterABC
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject


class ConfidenceObjectsFilter(FilterABC[DetectedObject]):
    """Keep only objects for which we are confident enough."""

    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold

    def validate_single(self, obj: DetectedObject) -> bool:
        return obj.confidence >= self.confidence_threshold
