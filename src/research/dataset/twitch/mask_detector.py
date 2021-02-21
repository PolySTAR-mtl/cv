from pathlib import Path

import numpy as np

from polystar.models.image import Image, load_image

DIR_PATH = Path(__file__).parent


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


robot_view_mask_hd = Mask(DIR_PATH / "mask_robot_view_hd.jpg", 20)
aerial_view_mask_red_hd = Mask(DIR_PATH / "mask_aerial_red_hd.jpg", 15)
aerial_view_mask_red_2_hd = Mask(DIR_PATH / "mask_aerial_red_2_hd.jpg", 15)
aerial_view_mask_blue_hd = Mask(DIR_PATH / "mask_aerial_blue_hd.jpg", 15)
aerial_view_mask_blue_2_hd = Mask(DIR_PATH / "mask_aerial_blue_2_hd.jpg", 15)
bonus_view_mask_hd = Mask(DIR_PATH / "mask_bonus.jpg", 20)
bonus_2_view_mask_hd = Mask(DIR_PATH / "mask_bonus_2.jpg", 20)
bonus_3_view_mask_hd = Mask(DIR_PATH / "mask_bonus_3.jpg", 20)


def is_aerial_view(image: Image) -> bool:
    return (
        aerial_view_mask_red_hd.match(image)
        or aerial_view_mask_red_2_hd.match(image)
        or aerial_view_mask_blue_hd.match(image)
        or aerial_view_mask_blue_2_hd.match(image)
    )


def has_bonus_icon(image: Image) -> bool:
    return bonus_view_mask_hd.match(image) or bonus_2_view_mask_hd.match(image) or bonus_3_view_mask_hd.match(image)


if __name__ == "__main__":
    has_bonus_icon(
        load_image(
            Path("/Users/cytadel/polystar/cv-code/dataset/twitch/robots-views/470152932/470152932-frame-007460.jpg")
        )
    )
