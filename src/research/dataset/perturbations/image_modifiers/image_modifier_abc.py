from abc import ABC, abstractmethod

from polystar.models.image import Image


class ImageModifierABC(ABC):
    @abstractmethod
    def modify(self, image: Image, intensity: float) -> Image:
        pass
