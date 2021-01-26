from pathlib import Path
from typing import Generic, Iterable, List, Tuple

from polystar.filters.exclude_filter import ExcludeFilter
from polystar.filters.filter_abc import FilterABC
from polystar.filters.pass_through_filter import PassThroughFilter
from polystar.models.image import FileImage
from research.armors.dataset.armor_value_dataset_cache import ArmorValueDatasetCache
from research.armors.dataset.armor_value_target_factory import ArmorValueTargetFactory
from research.common.datasets.dataset import Dataset
from research.common.datasets.image_file_dataset_builder import DirectoryDatasetBuilder
from research.common.datasets.lazy_dataset import TargetT
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.dataset.cleaning.dataset_changes import DatasetChanges


class ExcludeFilesFilter(ExcludeFilter[Path]):
    def validate_single(self, path: Path) -> bool:
        return path.name not in self.to_remove


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

    def default_datasets(
        self, include_dji: bool = True
    ) -> Tuple[List[Dataset[FileImage, TargetT]], List[Dataset[FileImage, TargetT]], List[Dataset[FileImage, TargetT]]]:
        return (
            self.default_train_datasets() if include_dji else self.twitch_train_datasets(),
            self.default_validation_datasets(),
            self.default_test_datasets(),
        )

    def default_test_datasets(self) -> List[Dataset[FileImage, TargetT]]:
        return self.from_roco_datasets(ROCODatasetsZoo.DEFAULT_TEST_DATASETS)

    def default_validation_datasets(self) -> List[Dataset[FileImage, TargetT]]:
        return self.from_roco_datasets(ROCODatasetsZoo.DEFAULT_VALIDATION_DATASETS)

    def default_train_datasets(self) -> List[Dataset[FileImage, TargetT]]:
        return self.from_roco_datasets(ROCODatasetsZoo.DEFAULT_TRAIN_DATASETS)

    def twitch_train_datasets(self) -> List[Dataset[FileImage, TargetT]]:
        return self.from_roco_datasets(ROCODatasetsZoo.TWITCH_TRAIN_DATASETS)

    # FIXME signature inconsistency across methods
    def from_roco_datasets(self, roco_datasets: Iterable[ROCODatasetBuilder]) -> List[Dataset[FileImage, TargetT]]:
        return [self.from_roco_dataset(roco_dataset).to_file_images().build() for roco_dataset in roco_datasets]

    def from_roco_dataset(self, roco_dataset_builder: ROCODatasetBuilder) -> DirectoryDatasetBuilder[TargetT]:
        cache_dir = roco_dataset_builder.main_dir / self.task_name
        dataset_name = roco_dataset_builder.name

        ArmorValueDatasetCache(
            roco_dataset_builder, cache_dir, dataset_name, self.target_factory
        ).generate_or_download_if_needed()

        return (
            DirectoryDatasetBuilder(cache_dir, self.target_factory.from_file, dataset_name)
            .filter_targets(self.targets_filter)
            .filter_examples(ExcludeFilesFilter(DatasetChanges(cache_dir).invalidated))
        )
