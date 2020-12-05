from dataclasses import dataclass

import cv2
from numpy.core._multiarray_umath import array, ndarray

from polystar.common.models.image import Image
from polystar.common.pipeline.pipe_abc import PipeABC


@dataclass
class Histogram2D(PipeABC):
    bins: int = 8

    def transform_single(self, image: Image) -> ndarray:
        return array([self._channel_hist(image, channel) for channel in range(3)]).ravel()

    def _channel_hist(self, image: Image, channel: int) -> ndarray:
        hist = cv2.calcHist([image], [channel], None, [self.bins], [0, 256], accumulate=False).ravel()
        return hist / hist.sum()

    @property
    def name(self) -> str:
        return "hist"
