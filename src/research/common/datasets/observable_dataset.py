from typing import Callable, Iterator, Tuple

from research.common.datasets.filter_dataset import ExampleU, TargetU
from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT


class ObservableDataset(LazyDataset[ExampleT, TargetT]):
    def __init__(
        self,
        source: LazyDataset[ExampleT, TargetT],
        example_observable: Callable[[ExampleT], None],
        target_observable: Callable[[TargetT], None],
    ):
        self.target_observable = target_observable
        self.example_observable = example_observable
        self.source = source
        super().__init__(source.name)

    def __iter__(self) -> Iterator[Tuple[ExampleU, TargetU, str]]:
        for example, target, name in self.source:
            self.example_observable(example)
            self.target_observable(target)
            yield example, target, name

    def __len__(self):
        return len(self.source)
