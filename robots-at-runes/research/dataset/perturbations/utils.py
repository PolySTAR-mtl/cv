from collections import Callable
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from polystar.common.models.image import Image

EXAMPLE_DIR = Path(__file__).parent / "examples"


def simple_perturbator_test(perturbator_function: Callable):
    rune_img = Image.from_path(EXAMPLE_DIR / "test.png")
    rune_perturbed = perturbator_function(rune_img, 1)
    cv2.imwrite(
        str(EXAMPLE_DIR / f"res_{perturbator_function.__name__}.png"), cv2.cvtColor(rune_perturbed, cv2.COLOR_RGB2BGR)
    )
    side_by_side_display = np.hstack((rune_img, rune_perturbed))
    h, w, _ = rune_img.shape
    plt.figure(figsize=(12, 6 * h / w))
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(side_by_side_display)
    plt.show()
