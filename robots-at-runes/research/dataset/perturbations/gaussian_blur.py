import cv2

from polystar.common.models.image import Image


def to_odd_number(number):
    return int(number // 2 * 2) - 1


def gaussian_blur(image: Image, intensity: float) -> Image:
    MAX_FACTOR = 0.015
    blur_factor = intensity * MAX_FACTOR
    width = image.shape[0]
    height = image.shape[1]

    x, y = to_odd_number(width * blur_factor), to_odd_number(height * blur_factor)
    image = cv2.GaussianBlur(image, (x, y), cv2.BORDER_DEFAULT)
    return image


if __name__ == "__main__":
    from research.dataset.perturbations.utils import simple_perturbator_test

    simple_perturbator_test(gaussian_blur)
