from abc import abstractmethod
from typing import TypeVar, Generic, Tuple, List

from polystar.common.models.image import Image
from research_common.dataset.roco_dataset import ROCODataset

T = TypeVar("T")


class ImageDatasetGenerator(Generic[T]):
    @abstractmethod
    def from_roco_dataset(self, dataset: ROCODataset) -> Tuple[List[Image], List[T]]:
        pass
