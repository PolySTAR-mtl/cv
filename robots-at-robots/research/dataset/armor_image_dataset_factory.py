import json
from abc import abstractmethod
from pathlib import Path
from typing import TypeVar, Tuple, Iterable

import cv2

from polystar.common.models.image import Image
from polystar.common.models.object import Armor
from polystar.common.utils.time import create_time_id
from research.dataset.armor_dataset_factory import ArmorDatasetFactory
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset
from research_common.image_pipeline_evaluation.image_dataset_generator import ImageDatasetGenerator

T = TypeVar("T")


class ArmorImageDatasetGenerator(ImageDatasetGenerator[T]):
    task_name: str

    def from_roco_dataset(self, dataset: DirectoryROCODataset) -> Iterable[Tuple[Image, T]]:
        if not (dataset.dataset_path / self.task_name / ".lock").exists():
            self._create_labelized_armor_images_from_roco(dataset)
        return self._get_images_paths_and_labels(dataset)

    def _create_labelized_armor_images_from_roco(self, dataset):
        dset_path = dataset.dataset_path / self.task_name
        dset_path.mkdir(exist_ok=True)
        for (armor_img, armor, k, path) in ArmorDatasetFactory.from_dataset(dataset):
            label = self._label_from_armor_info(armor, k, path)
            cv2.imwrite(str(dset_path / f"{path.stem}-{k}-{label}.jpg"), cv2.cvtColor(armor_img, cv2.COLOR_RGB2BGR))
        (dataset.dataset_path / self.task_name / ".lock").write_text(
            json.dumps({"version": "0.0", "date": create_time_id()})
        )

    def _get_images_paths_and_labels(self, dataset: DirectoryROCODataset) -> Iterable[Tuple[Image, T]]:
        return (
            (image_path, self._label_from_filepath(image_path))
            for image_path in (dataset.dataset_path / self.task_name).glob("*.jpg")
            if self._valid_label(self._label_from_filepath(image_path))
        )

    def _label_from_filepath(self, image_path: Path) -> T:
        return self._label_from_str(image_path.stem.split("-")[-1])

    @abstractmethod
    def _label_from_armor_info(self, armor: Armor, k: int, path: Path) -> T:
        pass

    def _valid_label(self, label: T) -> bool:
        return True

    def _label_from_str(self, label: str) -> T:
        return label
