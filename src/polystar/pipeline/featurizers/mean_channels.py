from dataclasses import dataclass
from nptyping import Array

from polystar.models.image import Image
from polystar.pipeline.pipe_abc import PipeABC


@dataclass
class MeanChannels(PipeABC):
    def transform_single(self, image: Image) -> Array[float, float, float]:
        return image.mean(axis=(0, 1))