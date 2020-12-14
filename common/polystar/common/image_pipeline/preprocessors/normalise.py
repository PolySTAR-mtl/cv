from polystar.common.models.image import Image
from polystar.common.pipeline.pipe_abc import PipeABC


class Normalise(PipeABC):
    def transform_single(self, image: Image) -> Image:
        return image / 255
