from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
from cv2.cv2 import VideoCapture

from polystar.common.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.common.models.image import Image


@dataclass
class CV2FrameGeneratorABC(FrameGeneratorABC, ABC):

    _cap: VideoCapture = field(init=False, repr=False)

    def __enter__(self):
        self._cap = cv2.VideoCapture(*self._capture_params())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cap.release()

    def generate(self) -> Iterable[Image]:
        with self:
            while 1:
                is_open, frame = self._cap.read()
                if not is_open:
                    return
                yield frame

    @abstractmethod
    def _capture_params(self) -> Iterable[Any]:
        pass
