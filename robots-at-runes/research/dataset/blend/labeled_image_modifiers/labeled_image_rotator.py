from dataclasses import dataclass

import cv2
import numpy as np
from imutils import rotate_bound

from polystar.common.models.image import Image
from research.dataset.blend.labeled_image_modifiers.labeled_image_modifier_abc import LabeledImageModifierABC
from research.dataset.labeled_image import PointOfInterest


@dataclass
class LabeledImageRotator(LabeledImageModifierABC):
    max_angle_degrees: float

    def _generate_modified_image(self, image: Image, angle: float) -> Image:
        return rotate_bound(image, angle)

    def _generate_modified_poi(
        self, poi: PointOfInterest, original_image: Image, new_image: Image, angle_degrees: float
    ) -> PointOfInterest:
        angle_rads = np.deg2rad(angle_degrees)
        sin, cos = np.sin(angle_rads), np.cos(angle_rads)
        rotation_matrix = np.array([[cos, -sin], [sin, cos]])
        prev_vector_to_center = np.array((poi.x - original_image.shape[1] / 2, poi.y - original_image.shape[0] / 2))
        new_vector_to_center = np.dot(rotation_matrix, prev_vector_to_center)
        return PointOfInterest(
            int(new_vector_to_center[0] + new_image.shape[1] / 2),
            int(new_vector_to_center[1] + new_image.shape[0] / 2),
            poi.label,
        )

    def _get_value_from_factor(self, factor: float) -> float:
        return self.max_angle_degrees * factor
