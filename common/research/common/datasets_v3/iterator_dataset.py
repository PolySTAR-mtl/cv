from typing import Iterator, Tuple

from research.common.datasets_v3.lazy_dataset import ExampleT, LazyDataset, TargetT


class IteratorDataset(LazyDataset[ExampleT, TargetT]):
    def __init__(self, iterator: Iterator[Tuple[ExampleT, TargetT, str]], name: str):
        super().__init__(name)
        self.iterator = iterator

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT, str]]:
        return self.iterator
