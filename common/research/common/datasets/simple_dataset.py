from typing import Iterable, Iterator, List, Tuple

from research.common.datasets.dataset import Dataset, ExampleT, TargetT


class SimpleDataset(Dataset[ExampleT, TargetT]):
    def __init__(self, examples: Iterable[ExampleT], targets: Iterable[TargetT], name: str):
        super().__init__(name)
        self._examples = list(examples)
        self._targets = list(targets)
        self.check_consistency()

    @property
    def examples(self) -> List[ExampleT]:
        return self._examples

    @property
    def targets(self) -> List[TargetT]:
        return self._targets

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT]]:
        return zip(self.examples, self.targets)

    def __len__(self):
        return len(self.examples)
