from dataclasses import dataclass
from typing import Tuple

from polystar.common.image_pipeline.image_featurizer.image_featurizer_abc import ImageFeaturizerABC
from polystar.common.models.image import Image


@dataclass
class MeanChannelsFeaturizer(ImageFeaturizerABC):
    def _featurize_one(self, image: Image) -> Tuple[float, float, float]:
        return image.mean(axis=(0, 1))

    def __str__(self) -> str:
        return "mean_channels"
