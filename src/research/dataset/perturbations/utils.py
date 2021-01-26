from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from polystar.models.image import load_image
from research.dataset.perturbations.image_modifiers.image_modifier_abc import ImageModifierABC

EXAMPLE_DIR = Path(__file__).parent / "examples"


def simple_modifier_demo(modifier: ImageModifierABC):
    rune_img = load_image(EXAMPLE_DIR / "test.png")
    rune_perturbed = modifier.modify(rune_img, 1)
    cv2.imwrite(str(EXAMPLE_DIR / f"res_{modifier}.png"), cv2.cvtColor(rune_perturbed, cv2.COLOR_RGB2BGR))
    side_by_side_display = np.hstack((rune_img, rune_perturbed))
    h, w, _ = rune_img.shape
    plt.figure(figsize=(12, 6 * h / w))
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(side_by_side_display)
    plt.show()
