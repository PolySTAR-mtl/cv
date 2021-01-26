from dataclasses import dataclass

import cv2
from numpy.core._multiarray_umath import array, ndarray

from polystar.models.image import Image
from polystar.pipeline.pipe_abc import PipeABC


@dataclass
class Histogram2D(PipeABC):
    bins: int = 8

    def transform_single(self, image: Image) -> ndarray:
        return array(
            [calculate_normalized_channel_histogram(image, channel, self.bins) for channel in range(3)]
        ).ravel()

    @property
    def name(self) -> str:
        return "hist"


def calculate_normalized_channel_histogram(image: Image, channel: int, bins: int) -> ndarray:
    hist = cv2.calcHist([image], [channel], None, [bins], [0, 256], accumulate=False).ravel()
    return hist / hist.sum()
