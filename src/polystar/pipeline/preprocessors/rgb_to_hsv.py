from dataclasses import dataclass

import cv2

from polystar.models.image import Image
from polystar.pipeline.pipe_abc import PipeABC


@dataclass
class RGB2HSV(PipeABC[Image, Image]):
    @property
    def name(self) -> str:
        return "hsv"

    def transform_single(self, image: Image) -> Image:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
