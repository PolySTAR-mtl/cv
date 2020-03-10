from abc import ABC, abstractmethod
from typing import Generator

from polystar.common.models.image import Image


class FrameGeneratorABC(ABC):
    @abstractmethod
    def generate(self) -> Generator[Image]:
        pass
