from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2

from polystar.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.models.image import Image


@dataclass
class CV2FrameGeneratorABC(FrameGeneratorABC, ABC):
    def generate(self) -> Iterable[Image]:
        _cap = self._open()
        while 1:
            is_open, frame = _cap.read()
            if not is_open:
                break
            yield frame
        _cap.release()

    def _open(self) -> cv2.VideoCapture:
        _cap = cv2.VideoCapture(*self._capture_params())
        _cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        assert _cap.isOpened()
        return _cap

    @abstractmethod
    def _capture_params(self) -> Iterable[Any]:
        pass
