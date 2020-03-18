import numpy as np

from polystar.common.models.object import Object
from polystar.common.target_pipeline.object_selectors.scored_object_selector_abc import ScoredObjectSelectorABC


class ClosestObjectSelector(ScoredObjectSelectorABC):
    """Take the object closest to the center of the image as a target"""

    def score(self, obj: Object, image: np.ndarray) -> float:
        d_x = obj.x + obj.w // 2 - image.shape[1] // 2
        d_y = obj.y + obj.h // 2 - image.shape[0] // 2
        return -(d_x ** 2 + d_y ** 2)
