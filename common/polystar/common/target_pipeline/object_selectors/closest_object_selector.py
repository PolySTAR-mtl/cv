import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.object_selectors.scored_object_selector_abc import ScoredObjectSelectorABC


class ClosestObjectSelector(ScoredObjectSelectorABC):
    """Take the object closest to the center of the image as a target"""

    def score(self, obj: DetectedObject, image: np.ndarray) -> float:
        d_x = obj.box.x + obj.box.w // 2 - image.shape[1] // 2
        d_y = obj.box.y + obj.box.h // 2 - image.shape[0] // 2
        return -(d_x ** 2 + d_y ** 2)
