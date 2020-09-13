from typing import Iterable, Iterator, Tuple

from research.common.datasets.dataset import (Dataset, ExampleT, LazyDataset,
                                              TargetT)


class UnionDataset(LazyDataset[ExampleT, TargetT]):
    def __init__(self, datasets: Iterable[Dataset[ExampleT, TargetT]], name: str = None):
        self.datasets = list(datasets)
        super().__init__(name or "_".join(d.name for d in self.datasets))

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT, str]]:
        for dataset in self.datasets:
            yield from dataset

    def __len__(self):
        return sum(map(len, self.datasets))
