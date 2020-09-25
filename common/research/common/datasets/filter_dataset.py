from typing import Iterator, Tuple, TypeVar

from polystar.common.filters.filter_abc import FilterABC
from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT

ExampleU = TypeVar("ExampleU")
TargetU = TypeVar("TargetU")


class FilterDataset(LazyDataset[ExampleT, TargetT]):
    def __init__(
        self,
        source: LazyDataset[ExampleT, TargetT],
        examples_filter: FilterABC[ExampleT],
        targets_filter: FilterABC[TargetT],
    ):
        super().__init__(source.name)
        self.targets_filter = targets_filter
        self.examples_filter = examples_filter
        self.source = source

    def __iter__(self) -> Iterator[Tuple[ExampleU, TargetU, str]]:
        for example, target, name in self.source:
            if self.examples_filter.validate_single(example) and self.targets_filter.validate_single(target):
                yield example, target, name
