from typing import Callable, Iterator, Tuple

from research.common.datasets.dataset_builder import ExampleU, TargetU
from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT


class TransformDataset(LazyDataset[ExampleU, TargetU]):
    def __init__(
        self,
        source: LazyDataset[ExampleT, TargetT],
        example_transformer: Callable[[ExampleT], ExampleU],
        target_transformer: Callable[[TargetT], TargetU],
    ):
        self.target_transformer = target_transformer
        self.example_transformer = example_transformer
        self.source = source
        super().__init__(source.name)

    def __iter__(self) -> Iterator[Tuple[ExampleU, TargetU, str]]:
        for example, target, name in self.source:
            yield self.example_transformer(example), self.target_transformer(target), name

    def __len__(self):
        return len(self.source)
