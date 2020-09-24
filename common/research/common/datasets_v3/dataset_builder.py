from typing import Callable, Generic, Iterable

from polystar.common.filters.filter_abc import FilterABC
from polystar.common.filters.pass_through_filter import PassThroughFilter
from polystar.common.utils.misc import identity
from research.common.datasets_v3.dataset import Dataset
from research.common.datasets_v3.filter_dataset import ExampleU, FilterDataset, TargetU
from research.common.datasets_v3.lazy_dataset import ExampleT, LazyDataset, TargetT
from research.common.datasets_v3.transform_dataset import TransformDataset


class DatasetBuilder(Generic[ExampleT, TargetT]):
    def __init__(self, dataset: LazyDataset[ExampleT, TargetT]):
        self.dataset = dataset
        self._built = False

    def build_lazy(self) -> LazyDataset[ExampleT, TargetT]:
        assert not self._built
        self._built = True
        return self.dataset

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

    @property
    def name(self) -> str:
        return self.dataset.name

    @name.setter
    def name(self, name: str):
        self.dataset.name = name
