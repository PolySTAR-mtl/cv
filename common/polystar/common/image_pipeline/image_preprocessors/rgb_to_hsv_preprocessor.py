from dataclasses import dataclass

import cv2

from polystar.common.image_pipeline.image_preprocessors.image_preprocessor_abc import ImagePreprocessorABC
from polystar.common.models.image import Image


@dataclass
class RGB2HSVPreprocessor(ImagePreprocessorABC):
    def _preprocess_one(self, img: Image) -> Image:
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    def __str__(self) -> str:
        return "hsv"
