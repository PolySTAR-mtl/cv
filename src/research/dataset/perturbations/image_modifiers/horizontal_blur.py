from dataclasses import dataclass

import cv2
import numpy as np

from polystar.models.image import Image
from research.dataset.perturbations.image_modifiers.image_modifier_abc import ImageModifierABC


@dataclass
class HorizontalBlurrer(ImageModifierABC):
    max_kernel_size: int = 11

    def modify(self, image: Image, intensity: float) -> Image:
        kernel = self._make_zero_kernel(intensity)
        self._fill_middle_row_with_ones(kernel)
        self._normalize_kernel(kernel)
        return self._apply_kernel(image, kernel)

    def _make_zero_kernel(self, intensity: float) -> np.ndarray:
        kernel_size = int(self.max_kernel_size * intensity) + 1
        return np.zeros((kernel_size, kernel_size))

    @staticmethod
    def _fill_middle_row_with_ones(kernel: np.ndarray):
        kernel[len(kernel) // 2, :] = 1

    @staticmethod
    def _normalize_kernel(kernel: np.ndarray):
        kernel /= len(kernel)

    @staticmethod
    def _apply_kernel(image: Image, kernel: np.ndarray) -> Image:
        return cv2.filter2D(image, -1, kernel)


if __name__ == "__main__":
    from research.dataset.perturbations.utils import simple_modifier_demo

    simple_modifier_demo(HorizontalBlurrer())
