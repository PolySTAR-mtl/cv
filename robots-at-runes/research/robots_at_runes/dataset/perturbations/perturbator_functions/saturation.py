import cv2
import numpy as np

from polystar.common.models.image import Image


def saturation(image: Image, intensity: float) -> Image:
    MAX_SATURATION = 0.6
    saturation_factor = 1 + MAX_SATURATION * intensity
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    (h, s, v) = cv2.split(image_hsv)
    new_s = np.clip((s.astype(np.uint16) * saturation_factor), 0, 255).astype(np.uint8)
    new_image_hsv = cv2.merge([h, new_s, v])
    image_rgb = cv2.cvtColor(new_image_hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
    return image_rgb


if __name__ == "__main__":
    from research.robots_at_runes.dataset.perturbations.utils import simple_perturbator_test

    simple_perturbator_test(saturation)
