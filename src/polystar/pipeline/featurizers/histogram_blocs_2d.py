from dataclasses import dataclass
from itertools import chain
from typing import Iterable

from numpy import array_split
from numpy.core._multiarray_umath import array, ndarray

from polystar.models.image import Image
from polystar.pipeline.featurizers.histogram_2d import calculate_normalized_channel_histogram
from polystar.pipeline.pipe_abc import PipeABC


@dataclass
class HistogramBlocs2D(PipeABC):
    bins: int = 8
    rows: int = 2
    cols: int = 3

    def transform_single(self, image: Image) -> ndarray:
        return array(
            [
                calculate_normalized_channel_histogram(bloc, channel, self.bins)
                for channel in range(3)
                for bloc in _split_images_in_blocs(image, self.rows, self.cols)
            ]
        ).ravel()


def _split_images_in_blocs(image: Image, n_rows: int, n_cols: int) -> Iterable[Image]:
    return chain.from_iterable(array_split(column, n_rows, axis=0) for column in array_split(image, n_cols, axis=1))
