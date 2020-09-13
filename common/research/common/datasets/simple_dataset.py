from typing import Iterable, Iterator, List, Tuple

from research.common.datasets.dataset import Dataset, ExampleT, TargetT


class SimpleDataset(Dataset[ExampleT, TargetT]):
    def __init__(self, examples: Iterable[ExampleT], targets: Iterable[TargetT], names: Iterable[str], name: str):
        super().__init__(name)
        self._examples = list(examples)
        self._targets = list(targets)
        self._names = list(names)
        self.check_consistency()

    @property
    def examples(self) -> List[ExampleT]:
        return self._examples

    @property
    def targets(self) -> List[TargetT]:
        return self._targets

    @property
    def names(self) -> List[TargetT]:
        return self._names

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT, str]]:
        return zip(self.examples, self.targets, self.names)

    def __len__(self):
        return len(self.examples)
