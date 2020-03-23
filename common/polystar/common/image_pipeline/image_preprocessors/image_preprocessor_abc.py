from abc import ABC, abstractmethod
from typing import List

from polystar.common.models.image import Image


class ImagePreprocessorABC(ABC):
    def preprocess(self, images: List[Image]) -> List[Image]:
        return [self._preprocess_one(img) for img in images]

    @abstractmethod
    def _preprocess_one(self, img: Image) -> Image:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
