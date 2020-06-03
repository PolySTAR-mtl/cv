import cv2
import numpy as np

from polystar.common.models.image import Image


def horizontal_blur(image: Image, intensity: float) -> Image:
    MAX_KERNEL_SIZE = 11
    kernel_size = int(MAX_KERNEL_SIZE * intensity) + 1
    # Fill kernel with zeros
    kernel_horizontal = np.zeros((kernel_size, kernel_size))
    # Fill middle row with ones
    kernel_horizontal[int(abs(kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize
    kernel_horizontal /= kernel_size

    return cv2.filter2D(image, -1, kernel_horizontal)


if __name__ == "__main__":
    from research.dataset.perturbations.utils import simple_perturbator_test

    simple_perturbator_test(horizontal_blur)
