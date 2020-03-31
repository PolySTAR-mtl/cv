from abc import abstractmethod
from typing import TypeVar, Generic, Tuple, List, Iterable

from polystar.common.models.image import Image
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset

T = TypeVar("T")


class ImageDatasetGenerator(Generic[T]):
    @abstractmethod
    def from_roco_dataset(self, dataset: DirectoryROCODataset) -> Tuple[List[Image], List[T]]:
        pass

    def from_roco_datasets(self, datasets: Iterable[DirectoryROCODataset]) -> Tuple[List[Image], List[T]]:
        images, labels = [], []
        for dataset in datasets:
            imgs, lbls = self.from_roco_dataset(dataset)
            images.extend(imgs)
            labels.extend(lbls)
        return images, labels
