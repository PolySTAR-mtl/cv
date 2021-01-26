from typing import Tuple

from cv2.cv2 import resize

from polystar.models.image import Image
from polystar.pipeline.pipe_abc import PipeABC
from polystar.utils.registry import registry


@registry.register()
class Resize(PipeABC):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def transform_single(self, image: Image) -> Image:
        return resize(image, self.size)
