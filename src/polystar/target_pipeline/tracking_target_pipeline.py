from dataclasses import dataclass
from typing import List

from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject
from polystar.target_pipeline.objects_trackers.objects_tracker_abc import ObjectsTrackerABC
from polystar.target_pipeline.target_pipeline import TargetPipeline


@dataclass
class TrackingTargetPipeline(TargetPipeline):
    tracker: ObjectsTrackerABC

    def _detect_all_objects(self, image) -> List[DetectedROCOObject]:
        return self.tracker.track_objects(super()._detect_all_objects(image), image)
