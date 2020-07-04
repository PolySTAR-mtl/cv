from dataclasses import dataclass

import numpy as np

from polystar.common.models.image import Image
from research.robots_at_runes.dataset.perturbations.image_modifiers.image_modifier_abc import ImageModifierABC


@dataclass
class BrightnessModifier(ImageModifierABC):
    max_beta: float = 10.0

    def modify(self, image: Image, intensity: float) -> Image:
        beta = self.max_beta * intensity
        perturbed_image = np.clip((image.astype(np.uint16) + beta), 0, 255).astype(np.uint8)
        return perturbed_image


if __name__ == "__main__":
    from research.robots_at_runes.dataset.perturbations.utils import simple_modifier_demo

    simple_modifier_demo(BrightnessModifier())
