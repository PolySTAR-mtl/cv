from pathlib import Path
from typing import Generic, List

from polystar.common.filters.filter_abc import FilterABC
from polystar.common.filters.pass_through_filter import PassThroughFilter
from research.common.datasets.dataset import Dataset
from research.common.datasets.image_file_dataset_builder import DirectoryDatasetBuilder
from research.common.datasets.lazy_dataset import TargetT
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.union_dataset import UnionDataset
from research.robots_at_robots.dataset.armor_value_dataset_cache import ArmorValueDatasetCache
from research.robots_at_robots.dataset.armor_value_target_factory import ArmorValueTargetFactory


class ArmorValueDatasetGenerator(Generic[TargetT]):
    def __init__(
        self,
        task_name: str,
        target_factory: ArmorValueTargetFactory[TargetT],
        targets_filter: FilterABC[TargetT] = None,
    ):
        self.target_factory = target_factory
        self.task_name = task_name
        self.targets_filter = targets_filter or PassThroughFilter()

    def from_roco_datasets(self, roco_datasets: List[ROCODatasetBuilder]) -> UnionDataset[Path, TargetT]:
        return UnionDataset(map(self.from_roco_dataset, roco_datasets))

    def from_roco_dataset(self, roco_dataset_builder: ROCODatasetBuilder) -> Dataset[Path, TargetT]:
        cache_dir = roco_dataset_builder.main_dir / self.task_name
        dataset_name = f"{roco_dataset_builder.name}_armor_{self.task_name}"

        cache = ArmorValueDatasetCache(roco_dataset_builder, cache_dir, dataset_name, self.target_factory)
        cache.generate_if_needed()

        return (
            DirectoryDatasetBuilder(cache_dir, self.target_factory.from_file, dataset_name)
            .filter_targets(self.targets_filter)
            .build()
        )
