from collections import deque
from typing import Optional

from dataclasses import dataclass

from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject


@dataclass
class ObjectTrack:
    new_object: Optional[DetectedObject]
    previous_object: Optional[DetectedObject]

    def merge(self) -> DetectedObject:

        if not self.previous_object:
            return self.new_object

        if not self.new_object:
            return self.previous_object

        self.new_object.previous_occurrences = self.previous_object.previous_occurrences
        self.previous_object.previous_occurrences = deque()
        self.new_object.previous_occurrences.append(self.previous_object)
