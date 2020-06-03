from dataclasses import dataclass, field
from itertools import chain
from typing import Any, List, Sequence

from polystar.common.image_pipeline.image_featurizer.identity_image_featurizer_abc import IdentityImageFeaturizer
from polystar.common.image_pipeline.image_featurizer.image_featurizer_abc import ImageFeaturizerABC
from polystar.common.image_pipeline.image_preprocessors.image_preprocessor_abc import ImagePreprocessorABC
from polystar.common.image_pipeline.models.model_abc import ModelABC
from polystar.common.models.image import Image


@dataclass
class ImagePipeline:
    image_preprocessors: List[ImagePreprocessorABC] = field(default_factory=list)
    image_featurizer: ImageFeaturizerABC = field(default_factory=IdentityImageFeaturizer)
    model: ModelABC = None
    custom_name: str = field(default="", repr=False)

    def predict(self, images: List[Image]) -> Sequence[Any]:
        preprocessed = self._preprocess(images)
        features = self.image_featurizer.featurize(preprocessed)
        return self.model.predict(features)

    def fit(self, images: List[Image], labels: List[Any]):
        preprocessed = self._preprocess(images)
        features = self.image_featurizer.fit(preprocessed, labels).featurize(preprocessed)
        self.model.fit(features, labels)

    def _preprocess(self, images: List[Image]) -> List[Image]:
        for preprocessor in self.image_preprocessors:
            images = preprocessor.preprocess(images)
        return images

    def __str__(self) -> str:
        return self.custom_name or "-".join(
            map(str, chain(self.image_preprocessors, [self.image_featurizer, self.model]))
        )
