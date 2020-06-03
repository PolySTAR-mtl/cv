import numpy as np

from polystar.common.models.image import Image


def contrast(image: Image, intensity: float) -> Image:
    ALPHA_FACTOR = 0.7
    MIN_ALPHA = 0.8
    alpha = MIN_ALPHA + ALPHA_FACTOR * intensity
    perturbed_image = np.clip((image.astype(np.uint16) * alpha), 0, 255).astype(np.uint8)
    return perturbed_image


if __name__ == "__main__":
    from research.dataset.perturbations.utils import simple_perturbator_test

    simple_perturbator_test(contrast)
