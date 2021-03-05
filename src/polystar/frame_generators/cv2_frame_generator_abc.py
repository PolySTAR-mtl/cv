from typing import Any, Iterable, Iterator

from cv2 import CAP_PROP_BUFFERSIZE, VideoCapture

from polystar.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.models.image import Image


class CV2FrameGenerator(FrameGeneratorABC):
    def __init__(self, *capture_params: Any):
        self.capture_params = capture_params or (0,)

    def __iter__(self) -> Iterator[Image]:
        return CV2Capture(self.capture_params)


class CV2Capture(Iterator[Image]):
    def __init__(self, capture_params: Iterable):
        self._cap = VideoCapture(*capture_params)
        assert self._cap.isOpened()
        self._cap.set(CAP_PROP_BUFFERSIZE, 0)

    def __next__(self) -> Image:
        success, frame = self._cap.read()

        if success:
            return frame

        raise StopIteration()

    def __del__(self):
        if self._cap.isOpened():
            self._cap.release()
