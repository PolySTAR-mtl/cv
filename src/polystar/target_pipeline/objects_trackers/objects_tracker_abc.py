from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List

from polystar.models.image import Image
from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject
from polystar.target_pipeline.objects_trackers.object_track import ObjectTrack


@dataclass
class ObjectsTrackerABC(ABC):
    n_steps_to_track: int

    tracked_objects: List[DetectedROCOObject] = field(init=False, default_factory=list)
    _step: int = field(init=False, default=0)

    def reset(self):
        self._step = 0

    def track_objects(self, objects: List[DetectedROCOObject], image: Image) -> List[DetectedROCOObject]:
        self._set_steps_of_new_objects(objects)
        self._loose_too_old_objects()
        tracks = self._link_to_previous_objects(objects, image)
        self.update_from_tracks(tracks)
        return self.tracked_objects

    def _set_steps_of_new_objects(self, objects):
        self._step += 1
        for obj in objects:
            obj.step_of_detection = self._step

    def _loose_too_old_objects(self):
        min_step_required = self._step - self.n_steps_to_track
        self.tracked_objects = [obj for obj in self.tracked_objects if obj.step_of_detection >= min_step_required]
        for obj in self.tracked_objects:
            if obj.previous_occurrences and obj.previous_occurrences[0].step_of_detection < min_step_required:
                obj.previous_occurrences.popleft()

    @abstractmethod
    def _link_to_previous_objects(self, objects: List[DetectedROCOObject], image: Image) -> Iterable[ObjectTrack]:
        pass

    def update_from_tracks(self, tracks: Iterable[ObjectTrack]):
        self.tracked_objects = [track.merge() for track in tracks]
