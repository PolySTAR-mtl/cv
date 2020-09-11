from itertools import chain
from typing import Iterable, Iterator, List, Tuple

from polystar.common.models.image import Image
from research.common.datasets.image_dataset import ImageDataset, TargetT


class UnionDataset(ImageDataset[TargetT]):
    def __init__(self, datasets: Iterable[ImageDataset[TargetT]], name: str):
        super().__init__(name)
        self.datasets = list(datasets)

    def __iter__(self) -> Iterator[Tuple[Image, TargetT]]:
        for dataset in self.datasets:
            yield from dataset

    @property
    def targets(self) -> List[TargetT]:
        return list(chain(*(dataset.targets for dataset in self.datasets)))
