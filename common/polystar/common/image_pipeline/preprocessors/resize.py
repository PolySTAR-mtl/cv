from typing import Tuple

from cv2.cv2 import resize

from polystar.common.models.image import Image
from polystar.common.pipeline.pipe_abc import PipeABC


class Resize(PipeABC):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def transform_single(self, image: Image) -> Image:
        return resize(image, self.size)
