from typing import Iterator, Tuple

from polystar.utils.iterable_utils import shuffle_iterable
from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT


class ShuffleDataset(LazyDataset):
    def __init__(self, source: LazyDataset[ExampleT, TargetT]):
        super().__init__(source.name)
        self.source = source

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT]]:
        return iter(shuffle_iterable(self.source))
