from typing import Iterator, List, Tuple

from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT


class Dataset(LazyDataset[ExampleT, TargetT]):
    def __init__(self, examples: List[ExampleT], targets: List[TargetT], names: List[str], name: str):
        super().__init__(name)
        self.names = names
        self.targets = targets
        self.examples = examples

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT, str]]:
        return zip(self.examples, self.targets, self.names)

    def __len__(self):
        return len(self.examples)
