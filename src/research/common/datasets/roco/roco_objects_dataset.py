from typing import Iterator, Tuple

from polystar.models.image import Image
from polystar.models.roco_object import Armor, ROCOObject
from research.common.datasets.lazy_dataset import LazyDataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset import LazyROCODataset


class ROCOObjectsDataset(LazyDataset[Image, ROCOObject]):
    def __init__(self, dataset: LazyROCODataset):
        super().__init__(f"{dataset.name}_objects")
        self.roco_dataset = dataset

    def __iter__(self) -> Iterator[Tuple[Image, ROCOObject, str]]:
        for image, annotation, name in self.roco_dataset:
            yield from self._generate_from_single(image, annotation, name)

    @staticmethod
    def _generate_from_single(
        image: Image, annotation: ROCOAnnotation, name: str
    ) -> Iterator[Tuple[Image, Armor, str]]:
        for i, obj in enumerate(annotation.objects):
            croped_img = image[obj.box.y1 : obj.box.y2, obj.box.x1 : obj.box.x2]
            yield croped_img, obj, f"{name}-{i}"
