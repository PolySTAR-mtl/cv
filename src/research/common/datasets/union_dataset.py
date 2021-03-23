from typing import Iterable, Iterator, List, Tuple

from research.common.datasets.dataset import Dataset
from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT


class UnionLazyDataset(LazyDataset[ExampleT, TargetT]):
    def __init__(self, datasets: Iterable[LazyDataset[ExampleT, TargetT]], name: str = None):
        self.datasets = list(datasets)
        super().__init__(name or name_from_union(self.datasets))

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT, str]]:
        for dataset in self.datasets:
            yield from dataset

    def __len__(self):
        return sum(map(len, self.datasets))


class UnionDataset(Dataset[ExampleT, TargetT]):
    def __init__(self, datasets: Iterable[Dataset[ExampleT, TargetT]], name: str = None):
        self.datasets = list(datasets)
        super().__init__(
            sum((d.examples for d in self.datasets), []),
            sum((d.targets for d in self.datasets), []),
            sum((d.names for d in self.datasets), []),
            name or name_from_union(self.datasets),
        )


def name_from_union(datasets: List[LazyDataset]):
    return "_".join(d.name for d in datasets)
