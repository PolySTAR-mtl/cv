from dataclasses import dataclass
from typing import List

import numpy as np

from polystar.common.image_pipeline.image_pipeline import ImagePipeline
from polystar.common.image_pipeline.models.classifier_model_abc import ClassifierModelABC
from polystar.common.models.image import Image


@dataclass
class ClassifierImagePipeline(ImagePipeline):
    model: ClassifierModelABC = None

    def predict_proba(self, images: List[Image]) -> np.ndarray:
        preprocessed = self._preprocess(images)
        features = self.image_featurizer.featurize(preprocessed)
        return self.model.predict_proba(features)
