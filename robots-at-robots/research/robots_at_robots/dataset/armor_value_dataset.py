import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, List, TypeVar

from polystar.common.models.image import Image, save_image
from polystar.common.models.object import Armor
from polystar.common.utils.time import create_time_id
from polystar.common.utils.tqdm import smart_tqdm
from research.common.datasets.dataset import Dataset
from research.common.datasets.image_dataset import ImageDirectoryDataset
from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.common.datasets.union_dataset import UnionDataset
from research.robots_at_robots.dataset.armor_dataset_factory import \
    ArmorDatasetFactory

ValueT = TypeVar("ValueT")


class ArmorValueDirectoryDataset(Generic[ValueT], ImageDirectoryDataset[ValueT], ABC):
    def target_from_image_file(self, image_file: Path) -> ValueT:
        return self._value_from_str(image_file.stem.split("-")[-1])

    @abstractmethod
    def _value_from_str(self, label: str) -> ValueT:
        pass


class ArmorValueDatasetGenerator(Generic[ValueT], ABC):
    VERSION: ClassVar[str] = "1.0"

    def __init__(self, task_name: str):
        self.task_name = task_name

    def from_roco_datasets(self, roco_datasets: List[DirectoryROCODataset]) -> UnionDataset[Path, ValueT]:
        return UnionDataset(map(self.from_roco_dataset, roco_datasets))

    def from_roco_dataset(self, roco_dataset: DirectoryROCODataset) -> Dataset[Path, ValueT]:
        self._generate_if_absent(roco_dataset)
        return self.from_directory_and_name(
            roco_dataset.main_dir / self.task_name, f"{roco_dataset.name}_armor_{self.task_name}"
        )

    @abstractmethod
    def from_directory_and_name(self, directory: Path, name: str) -> Dataset[Path, ValueT]:
        pass

    def _generate_if_absent(self, roco_dataset: DirectoryROCODataset):
        if self._exists_and_is_valid(roco_dataset):
            return
        self._generate(roco_dataset)

    def _task_dir(self, roco_dataset: DirectoryROCODataset) -> Path:
        return roco_dataset.main_dir / self.task_name

    def _generate(self, roco_dataset: DirectoryROCODataset):
        armor_dataset = self._make_dataset(roco_dataset)
        for image, target, name in smart_tqdm(
            armor_dataset, desc=f"Generating dataset {roco_dataset.name}_{self.task_name} ", unit="frame"
        ):
            save_image(image, self._task_dir(roco_dataset) / f"{name}-{target}.jpg")
        self._lock_file(roco_dataset).write_text(json.dumps({"version": self.VERSION, "date": create_time_id()}))

    def _exists_and_is_valid(self, roco_dataset: DirectoryROCODataset) -> bool:
        lock = self._lock_file(roco_dataset)
        return lock.exists() and json.loads(lock.read_text())["version"] == self.VERSION

    def _make_dataset(self, roco_dataset) -> Dataset[Image, ValueT]:
        return ArmorDatasetFactory(roco_dataset).make().transform_targets(self._value_from_armor)

    def _lock_file(self, roco_dataset: DirectoryROCODataset) -> Path:
        return self._task_dir(roco_dataset) / ".lock"

    @abstractmethod
    def _value_from_armor(self, armor: Armor) -> ValueT:
        pass
