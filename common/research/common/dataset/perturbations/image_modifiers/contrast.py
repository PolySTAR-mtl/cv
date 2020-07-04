from dataclasses import dataclass

import numpy as np

from polystar.common.models.image import Image
from research.common.dataset.perturbations.image_modifiers.image_modifier_abc import ImageModifierABC


@dataclass
class ContrastModifier(ImageModifierABC):
    min_coef: float = 0.8
    max_coef: float = 1.5

    def modify(self, image: Image, intensity: float) -> Image:
        coef = self.min_coef + (self.max_coef - self.min_coef) * intensity
        perturbed_image = np.clip((image.astype(np.uint16) * coef), 0, 255).astype(np.uint8)
        return perturbed_image


if __name__ == "__main__":
    from research.common.dataset.perturbations.utils import simple_modifier_demo

    simple_modifier_demo(ContrastModifier())
