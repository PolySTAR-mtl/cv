from dataclasses import dataclass

import cv2

from polystar.common.models.image import Image
from research.common.dataset.perturbations.image_modifiers.image_modifier_abc import ImageModifierABC


@dataclass
class GaussianBlurrer(ImageModifierABC):
    max_factor: float = 0.015

    def modify(self, image: Image, intensity: float) -> Image:
        blur_factor = intensity * self.max_factor
        width, height, *_ = image.shape

        x, y = _to_odd_number(width * blur_factor), _to_odd_number(height * blur_factor)
        image = cv2.GaussianBlur(image, (x, y), cv2.BORDER_DEFAULT)
        return image


def _to_odd_number(number):
    return int(number // 2 * 2) - 1


if __name__ == "__main__":
    from research.common.dataset.perturbations.utils import simple_modifier_demo

    simple_modifier_demo(GaussianBlurrer())
