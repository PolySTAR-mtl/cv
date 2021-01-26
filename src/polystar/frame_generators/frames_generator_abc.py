from abc import ABC, abstractmethod
from typing import Iterable

from polystar.models.image import Image


class FrameGeneratorABC(ABC):
    @abstractmethod
    def generate(self) -> Iterable[Image]:
        pass
