from abc import abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, Tuple, List, Iterable

from polystar.common.models.image import Image
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset

T = TypeVar("T")


class ImageDatasetGenerator(Generic[T]):
    def from_roco_datasets(
        self, datasets: Iterable[DirectoryROCODataset]
    ) -> Tuple[List[Path], List[Image], List[T], List[int]]:
        images_path, images, labels, dataset_sizes = [], [], [], []
        for dataset in datasets:
            prev_total_size = len(images)
            for img_path, label in self.from_roco_dataset(dataset):
                images_path.append(img_path)
                images.append(Image.from_path(img_path))
                labels.append(label)
            dataset_sizes.append(len(images) - prev_total_size)
        return images_path, images, labels, dataset_sizes

    @abstractmethod
    def from_roco_dataset(self, dataset: DirectoryROCODataset) -> Iterable[Tuple[Path, T]]:
        pass
