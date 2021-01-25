from random import shuffle
from typing import Iterator, Tuple

from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT


class ShuffleDataset(LazyDataset):
    def __init__(self, source: LazyDataset[ExampleT, TargetT]):
        super().__init__(source.name)
        self.source = source

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT]]:
        data = list(self.source)
        shuffle(data)
        return iter(data)
