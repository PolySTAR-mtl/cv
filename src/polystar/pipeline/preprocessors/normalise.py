from polystar.models.image import Image
from polystar.pipeline.pipe_abc import PipeABC
from polystar.utils.registry import registry


@registry.register()
class Normalise(PipeABC):
    def transform_single(self, image: Image) -> Image:
        return image / 255
