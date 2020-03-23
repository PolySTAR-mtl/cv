from dataclasses import dataclass

from polystar.common.image_pipeline.image_featurizer.image_featurizer_abc import ImageFeaturizerABC
from polystar.common.models.image import Image


@dataclass
class IdentityImageFeaturizer(ImageFeaturizerABC):
    def _featurize_one(self, img: Image) -> Image:
        return img

    def __str__(self) -> str:
        return "_"
