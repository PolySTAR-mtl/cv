from abc import ABC, abstractmethod
from typing import List, Any

from polystar.common.models.image import Image


class ImageFeaturizerABC(ABC):
    def fit(self, images: List[Image], labels: List[Any]) -> "ImageFeaturizerABC":
        return self

    def featurize(self, images: List[Image]) -> List[Any]:
        return [self._featurize_one(img) for img in images]

    @abstractmethod
    def _featurize_one(self, image: Image) -> Any:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
