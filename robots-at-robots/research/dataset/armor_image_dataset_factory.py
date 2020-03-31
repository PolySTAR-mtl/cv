from abc import abstractmethod
from pathlib import Path
from typing import TypeVar, Tuple, List, Iterable

from polystar.common.models.image import Image
from polystar.common.models.object import ArmorColor
from research.dataset.armor_dataset_factory import ArmorDatasetFactory
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset
from research_common.image_pipeline_evaluation.image_dataset_generator import ImageDatasetGenerator

T = TypeVar("T")


class ArmorImageDatasetGenerator(ImageDatasetGenerator[T]):
    def from_roco_dataset(self, dataset: DirectoryROCODataset) -> Iterable[Tuple[Image, T]]:
        for (armor_img, color, digit, k, path) in ArmorDatasetFactory.from_dataset(dataset):
            label = self._label(color, digit, k, path)
            if self._valid_label(label):
                yield armor_img, label

    @abstractmethod
    def _label(self, color: ArmorColor, digit: int, k: int, path: Path) -> T:
        pass

    def _valid_label(self, label: T) -> bool:
        return True
