import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.object_selectors.scored_object_selector_abc import ScoredObjectSelectorABC


class ClosestObjectSelector(ScoredObjectSelectorABC):
    """Take the armor closest to the center of the image as a target"""

    def score(self, armor: DetectedArmor, image: np.ndarray) -> float:
        d_x = armor.box.x + armor.box.w // 2 - image.shape[1] // 2
        d_y = armor.box.y + armor.box.h // 2 - image.shape[0] // 2
        return -(d_x ** 2 + d_y ** 2)
