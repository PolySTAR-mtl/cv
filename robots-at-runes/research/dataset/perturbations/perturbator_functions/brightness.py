import numpy as np

from polystar.common.models.image import Image


def brightness(image: Image, intensity: float) -> Image:
    MAX_BETA = 10
    beta = MAX_BETA * intensity
    perturbed_image = np.clip((image.astype(np.uint16) + beta), 0, 255).astype(np.uint8)
    return perturbed_image


if __name__ == "__main__":
    from research.dataset.perturbations.utils import simple_perturbator_test

    simple_perturbator_test(brightness)
