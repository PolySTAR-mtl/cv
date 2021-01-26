from collections import deque
from dataclasses import dataclass
from typing import Optional

from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject


@dataclass
class ObjectTrack:
    new_object: Optional[DetectedROCOObject]
    previous_object: Optional[DetectedROCOObject]

    def merge(self) -> DetectedROCOObject:

        if not self.previous_object:
            return self.new_object

        if not self.new_object:
            return self.previous_object

        self.new_object.previous_occurrences = self.previous_object.previous_occurrences
        self.previous_object.previous_occurrences = deque()
        self.new_object.previous_occurrences.append(self.previous_object)
