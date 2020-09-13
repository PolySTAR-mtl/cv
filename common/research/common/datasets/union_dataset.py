from typing import Iterable, Iterator, Tuple

from polystar.common.models.image import Image
from research.common.datasets.dataset import ExampleT, LazyDataset, TargetT
from research.common.datasets.image_dataset import ImageDataset


class UnionDataset(LazyDataset[ExampleT, TargetT]):
    def __init__(self, datasets: Iterable[ImageDataset[TargetT]], name: str):
        super().__init__(name)
        self.datasets = list(datasets)

    def __iter__(self) -> Iterator[Tuple[Image, TargetT]]:
        for dataset in self.datasets:
            yield from dataset

    def __len__(self):
        return sum(map(len, self.datasets))
