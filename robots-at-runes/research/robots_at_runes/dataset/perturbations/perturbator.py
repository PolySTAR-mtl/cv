from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from polystar.common.models.image import Image
from research.robots_at_runes.dataset.perturbations.image_modifiers.contrast import ContrastModifier
from research.robots_at_runes.dataset.perturbations.image_modifiers.gaussian_blur import GaussianBlurrer
from research.robots_at_runes.dataset.perturbations.image_modifiers.gaussian_noise import GaussianNoiser
from research.robots_at_runes.dataset.perturbations.image_modifiers.horizontal_blur import HorizontalBlurrer
from research.robots_at_runes.dataset.perturbations.image_modifiers.image_modifier_abc import ImageModifierABC


@dataclass
class ImagePerturbator:
    modifiers: List[ImageModifierABC]

    def perturbate(self, image: Image) -> Image:
        shuffle(self.modifiers)
        intensities = self._generate_intensities()
        for modifier, intensity in zip(self.modifiers, intensities):
            image = modifier.modify(image, intensity)
        return image

    def _generate_intensities(self) -> List[float]:
        intensities = np.random.random(len(self.modifiers))
        return intensities / intensities.sum()


if __name__ == "__main__":
    EXAMPLE_DIR = Path(__file__).parent / "examples"
    rune_img = Image.from_path(EXAMPLE_DIR / "test.png")
    perturbator = ImagePerturbator([ContrastModifier(), GaussianBlurrer(), GaussianNoiser(), HorizontalBlurrer()])
    rune_perturbed = perturbator.perturbate(rune_img)
    cv2.imwrite(str(EXAMPLE_DIR / "res_full_pipeline.png"), cv2.cvtColor(rune_perturbed, cv2.COLOR_RGB2BGR))
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(rune_perturbed)
    plt.show()
