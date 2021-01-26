from polystar.filters.filter_abc import FilterABC
from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject


class ConfidenceObjectsFilter(FilterABC[DetectedROCOObject]):
    """Keep only objects for which we are confident enough."""

    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold

    def validate_single(self, obj: DetectedROCOObject) -> bool:
        return obj.confidence >= self.confidence_threshold
