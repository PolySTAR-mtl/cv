from polystar.common.filters.filter_abc import FilterABC
from research.common.datasets.dataset import Dataset, ExampleT, TargetT
from research.common.datasets.simple_dataset import SimpleDataset


class FilteredTargetsDataset(SimpleDataset[ExampleT, TargetT]):
    def __init__(self, dataset: Dataset[ExampleT, TargetT], targets_filter: FilterABC[TargetT]):
        targets, examples, names = targets_filter.filter_with_siblings(
            list(dataset.targets), list(dataset.examples), list(dataset.names)
        )
        super().__init__(examples, targets, names, dataset.name)


class FilteredExamplesDataset(SimpleDataset[ExampleT, TargetT]):
    def __init__(self, dataset: Dataset[ExampleT, TargetT], examples_filter: FilterABC[ExampleT]):
        super().__init__(
            *examples_filter.filter_with_siblings(list(dataset.examples), list(dataset.targets), list(dataset.names)),
            dataset.name,
        )
