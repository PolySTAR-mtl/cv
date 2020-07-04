import numpy as np

from polystar.common.models.image import Image


def gaussian_noise(image: Image, intensity: float) -> Image:
    MAX_VARIANCE = 300
    mean = 0
    variance = MAX_VARIANCE * intensity
    sigma = variance ** 0.5
    row, column, ch = image.shape
    gaussian = np.random.normal(mean, sigma, (row, column, ch))
    perturbed_image = np.clip((image.astype(np.uint16) + gaussian), 0, 255).astype(np.uint8)
    return perturbed_image


if __name__ == "__main__":
    from research.robots_at_runes.dataset.perturbations.utils import simple_perturbator_test

    simple_perturbator_test(gaussian_noise)
