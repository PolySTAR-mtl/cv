from itertools import islice
from typing import Iterator, Tuple

from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT


class CappedDataset(LazyDataset):
    def __init__(self, source: LazyDataset[ExampleT, TargetT], n: int):
        super().__init__(source.name)
        self.n = n
        self.source = source

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT]]:
        return islice(self.source, self.n)
