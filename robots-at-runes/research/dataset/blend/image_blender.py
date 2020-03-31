from dataclasses import dataclass
from random import random
from typing import List, Tuple

import cv2
import numpy as np

from polystar.common.models.image import Image
from research.dataset.blend.labeled_image_modifiers.labeled_image_modifier_abc import LabeledImageModifierABC
from research.dataset.labeled_image import LabeledImage, PointOfInterest


@dataclass
class ImageBlender:
    background_size: Tuple[int, int]
    object_modifiers: List[LabeledImageModifierABC]

    def blend(self, background: Image, obj: LabeledImage) -> LabeledImage:
        obj = self._modify_object(obj)
        background = self._crop_background(background)
        x, y = self._generate_position_of_object(background.shape, obj.image.shape)
        return LabeledImage(
            image=self._blend_obj_on_background(background, obj.image, x, y),
            point_of_interests=[self._translate_poi(poi, x, y) for poi in obj.point_of_interests],
        )

    def _modify_object(self, obj: LabeledImage) -> LabeledImage:
        for modifier in self.object_modifiers:
            obj = modifier.randomly_modify(obj)
        return obj

    def _generate_position_of_object(
        self, background_shape: Tuple[int, int, int], obj_shape: Tuple[int, int, int]
    ) -> Tuple[int, int]:
        return (
            self._generate_position_of_object_amoung_axis(background_shape[1], obj_shape[1]),
            self._generate_position_of_object_amoung_axis(background_shape[0], obj_shape[0]),
        )

    @staticmethod
    def _generate_position_of_object_amoung_axis(background_size: int, obj_size: int) -> int:
        return int(random() * (background_size - obj_size))

    def _blend_obj_on_background(self, background: Image, obj_img: Image, x: int, y: int) -> Image:
        background_roi = background[y : y + obj_img.shape[0], x : x + obj_img.shape[1], :]
        mask = obj_img[:, :, 3]
        obj_img = cv2.cvtColor(obj_img, cv2.COLOR_RGBA2RGB)
        background_roi = background_roi.astype(np.float)
        obj_img = obj_img.astype(np.float)
        rv = background.copy()
        for i in range(3):
            rv[y : y + obj_img.shape[0], x : x + obj_img.shape[1], i] = (
                (~mask * background_roi[:, :, i] + mask * obj_img[:, :, i]) / 255
            ).astype(np.uint8)
        return rv

    @staticmethod
    def _translate_poi(poi: PointOfInterest, x: int, y: int) -> PointOfInterest:
        return PointOfInterest(poi.x + x, poi.y + y, poi.label)

    def _crop_background(self, background: Image) -> Image:
        h, w, _ = background.shape
        x, y = int(random() * (h - self.background_size[1])), int(random() * (w - self.background_size[0]))
        return background[y : y + self.background_size[1], x : x + self.background_size[0], :]


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    from research.dataset.blend.labeled_image_modifiers.labeled_image_rotator import LabeledImageRotator
    from research.dataset.blend.labeled_image_modifiers.labeled_image_scaler import LabeledImageScaler

    EXAMPLES_DIR = Path(__file__).parent / "examples"

    _obj = LabeledImage(
        cv2.cvtColor(cv2.imread(str(EXAMPLES_DIR / "logo.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA),
        PointOfInterest.from_annotation_file(EXAMPLES_DIR / "logo.xml"),
    )
    _bg = cv2.cvtColor(cv2.imread(str(EXAMPLES_DIR / "back1.jpg")), cv2.COLOR_BGR2RGB)

    _blender = ImageBlender(
        background_size=(1_280, 720), object_modifiers=[LabeledImageScaler(1.5), LabeledImageRotator(180)]
    )
    for i in range(10):
        res = _blender.blend(_bg, _obj)

        res.save(EXAMPLES_DIR, f"test_{i}")

        plt.imshow(res.image)
        for poi in res.point_of_interests:
            plt.plot([poi.x], [poi.y], f"{poi.label[0]}.")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
