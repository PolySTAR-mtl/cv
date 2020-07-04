from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import Callable, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from polystar.common.models.image import Image
from research.robots_at_runes.dataset.perturbations.perturbator_functions.contrast import contrast
from research.robots_at_runes.dataset.perturbations.perturbator_functions.gaussian_blur import gaussian_blur
from research.robots_at_runes.dataset.perturbations.perturbator_functions.gaussian_noise import gaussian_noise
from research.robots_at_runes.dataset.perturbations.perturbator_functions.horizontal_blur import horizontal_blur


@dataclass
class ImagePerturbator:
    perturbator_functions: List[Callable[[Image, float], Image]]

    def perturbate(self, image: Image) -> Image:
        shuffle(self.perturbator_functions)
        intensities = self._generate_intensities()
        for perturbator_function, intensity in zip(self.perturbator_functions, intensities):
            image = perturbator_function(image, intensity)
        return image

    def _generate_intensities(self) -> List[float]:
        intensities = np.random.random(len(self.perturbator_functions))
        return intensities / intensities.sum()


if __name__ == "__main__":
    EXAMPLE_DIR = Path(__file__).parent / "examples"
    rune_img = Image.from_path(EXAMPLE_DIR / "test.png")
    perturbator = ImagePerturbator([contrast, gaussian_blur, gaussian_noise, horizontal_blur])
    rune_perturbed = perturbator.perturbate(rune_img)
    cv2.imwrite(str(EXAMPLE_DIR / "res_full_pipeline.png"), cv2.cvtColor(rune_perturbed, cv2.COLOR_RGB2BGR))
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(rune_perturbed)
    plt.show()
