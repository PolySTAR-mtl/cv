from dataclasses import dataclass

import cv2

from polystar.models.image import Image
from research.runes.dataset.blend.labeled_image_modifiers.labeled_image_modifier_abc import LabeledImageModifierABC
from research.runes.dataset.labeled_image import PointOfInterest


@dataclass
class LabeledImageScaler(LabeledImageModifierABC):
    max_scale: float

    def _generate_modified_image(self, image: Image, scale: float) -> Image:
        return cv2.resize(
            image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA
        )

    def _generate_modified_poi(
        self, poi: PointOfInterest, original_image: Image, new_image: Image, scale: float
    ) -> PointOfInterest:
        return PointOfInterest(int(poi.x * scale), int(poi.y * scale), poi.label)

    def _get_value_from_factor(self, factor: float) -> float:
        intensity = (self.max_scale - 1) * abs(factor) + 1
        if factor > 0:
            return intensity
        return 1 / intensity
