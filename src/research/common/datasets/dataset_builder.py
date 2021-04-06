from typing import Callable, Generic, Iterable, Iterator, Tuple

from polystar.filters.filter_abc import FilterABC
from polystar.filters.pass_through_filter import PassThroughFilter
from polystar.utils.misc import identity
from research.common.datasets.dataset import Dataset
from research.common.datasets.filter_dataset import ExampleU, FilterDataset, TargetU
from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT
from research.common.datasets.observable_dataset import ObservableDataset
from research.common.datasets.shuffle_dataset import ShuffleDataset
from research.common.datasets.slice_dataset import SliceDataset
from research.common.datasets.transform_dataset import TransformDataset
from research.common.datasets.union_dataset import UnionLazyDataset


class DatasetBuilder(Generic[ExampleT, TargetT], Iterable[Tuple[ExampleT, TargetT, str]]):
    def __init__(self, dataset: LazyDataset[ExampleT, TargetT]):
        self.dataset = dataset
        self._built = False

    def build_lazy(self) -> LazyDataset[ExampleT, TargetT]:
        assert not self._built
        self._built = True
        return self.dataset

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT, str]]:
        return iter(self.build_lazy())

    def __len__(self):
        return len(self.dataset)

    def build(self) -> Dataset[ExampleT, TargetT]:
        assert not self._built
        self._built = True
        examples, targets, names = zip(*iter(self.dataset))
        return Dataset(list(examples), list(targets), list(names), self.name)

    def build_examples(self) -> Iterable[ExampleT]:
        assert not self._built
        self._built = True
        for ex, _, _ in self.dataset:
            yield ex

    def filter_examples(self, examples_filter: FilterABC[ExampleT]) -> "DatasetBuilder[ExampleT, TargetT]":
        self.dataset = FilterDataset(self.dataset, examples_filter, PassThroughFilter())
        return self

    def filter_targets(self, targets_filter: FilterABC[ExampleT]) -> "DatasetBuilder[ExampleT, TargetT]":
        self.dataset = FilterDataset(self.dataset, PassThroughFilter(), targets_filter)
        return self

    def transform_examples(
        self, example_transformer: Callable[[ExampleT], ExampleU]
    ) -> "DatasetBuilder[ExampleU, TargetT]":
        self.dataset = TransformDataset(self.dataset, example_transformer, identity)
        return self

    def transform_targets(
        self, target_transformer: Callable[[TargetT], TargetU]
    ) -> "DatasetBuilder[ExampleT, TargetU]":
        self.dataset = TransformDataset(self.dataset, identity, target_transformer)
        return self

    def apply_to_examples(self, example_observable: Callable[[ExampleT], None]) -> "DatasetBuilder[ExampleU, TargetT]":
        self.dataset = ObservableDataset(self.dataset, example_observable, identity)
        return self

    def apply_to_targets(self, target_observable: Callable[[TargetT], None]) -> "DatasetBuilder[ExampleT, TargetU]":
        self.dataset = ObservableDataset(self.dataset, identity, target_observable)
        return self

    def shuffle(self) -> "DatasetBuilder[ExampleT, TargetU]":
        self.dataset = ShuffleDataset(self.dataset)
        return self

    def cap(self, n: int) -> "DatasetBuilder[ExampleT, TargetT]":
        self.dataset = SliceDataset(self.dataset, stop=n)
        return self

    def skip(self, n: int) -> "DatasetBuilder[ExampleT, TargetT]":
        self.dataset = SliceDataset(self.dataset, start=n)
        return self

    def __or__(self, other: "DatasetBuilder[ExampleT, TargetT]"):
        self.dataset = UnionLazyDataset((self.dataset, other.dataset))
        return self

    @property
    def name(self) -> str:
        return self.dataset.name

    @name.setter
    def name(self, name: str):
        self.dataset.name = name
