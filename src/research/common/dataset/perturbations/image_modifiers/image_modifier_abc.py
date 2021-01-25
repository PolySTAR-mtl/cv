from abc import ABC, abstractmethod

from polystar.common.models.image import Image


class ImageModifierABC(ABC):
    @abstractmethod
    def modify(self, image: Image, intensity: float) -> Image:
        pass
