from pathlib import Path

import numpy as np

from polystar.models.image import Image, load_image

MASKS_DIR = Path(__file__).parent / "masks"


class Mask:
    def __init__(self, mask_file: Path, threshold: float):
        self._threshold = threshold
        mask_image = load_image(mask_file)
        self._mask_coordinates = np.where(mask_image.max(axis=-1) > 40)
        self._mask_values = mask_image[self._mask_coordinates].astype(np.int16)

    def match(self, image: Image) -> bool:
        value = np.abs(self._mask_values - image[self._mask_coordinates]).mean()
        # print(value)
        return value <= self._threshold


robot_view_mask_hd = Mask(MASKS_DIR / "mask_robot_view_hd.jpg", 20)
aerial_view_mask_red_hd = Mask(MASKS_DIR / "mask_aerial_red_hd.jpg", 15)
aerial_view_mask_red_2_hd = Mask(MASKS_DIR / "mask_aerial_red_2_hd.jpg", 15)
aerial_view_mask_blue_hd = Mask(MASKS_DIR / "mask_aerial_blue_hd.jpg", 15)
aerial_view_mask_blue_2_hd = Mask(MASKS_DIR / "mask_aerial_blue_2_hd.jpg", 15)
bonus_view_mask_hd = Mask(MASKS_DIR / "mask_bonus.jpg", 20)
bonus_2_view_mask_hd = Mask(MASKS_DIR / "mask_bonus_2.jpg", 20)
bonus_3_view_mask_hd = Mask(MASKS_DIR / "mask_bonus_3.jpg", 20)


def is_aerial_view(image: Image) -> bool:
    return (
        aerial_view_mask_red_hd.match(image)
        or aerial_view_mask_red_2_hd.match(image)
        or aerial_view_mask_blue_hd.match(image)
        or aerial_view_mask_blue_2_hd.match(image)
    )


def has_bonus_icon(image: Image) -> bool:
    return bonus_view_mask_hd.match(image) or bonus_2_view_mask_hd.match(image) or bonus_3_view_mask_hd.match(image)
