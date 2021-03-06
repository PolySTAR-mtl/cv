from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from polystar.models.image import Image


class FrameGeneratorABC(ABC, Iterable[Image]):
    @abstractmethod
    def __iter__(self) -> Iterator[Image]:
        pass
