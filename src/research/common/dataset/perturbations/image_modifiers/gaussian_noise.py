from dataclasses import dataclass

import numpy as np

from polystar.common.models.image import Image
from research.common.dataset.perturbations.image_modifiers.image_modifier_abc import ImageModifierABC


@dataclass
class GaussianNoiser(ImageModifierABC):
    max_variance: float = 300.0

    def modify(self, image: Image, intensity: float) -> Image:
        variance = self.max_variance * intensity
        sigma = variance ** 0.5
        row, column, ch = image.shape
        gaussian = np.random.normal(0, sigma, (row, column, ch))
        perturbed_image = np.clip((image.astype(np.uint16) + gaussian), 0, 255).astype(np.uint8)
        return perturbed_image


if __name__ == "__main__":
    from research.common.dataset.perturbations.utils import simple_modifier_demo

    simple_modifier_demo(GaussianNoiser())
