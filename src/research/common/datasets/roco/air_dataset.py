from typing import Iterator, Tuple

from polystar.models.image import Image
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC
from polystar.target_pipeline.objects_filters.type_object_filter import ARMORS_FILTER
from research.armors.dataset.armor_value_dataset_cache import DatasetCache
from research.common.datasets.lazy_dataset import LazyDataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset import LazyROCODataset
from research.dataset.improvement.zoom import crop_image_annotation


class AIRDataset(LazyDataset[Image, ROCOAnnotation]):
    """Armors In Robots"""

    def __init__(self, roco_dataset: LazyROCODataset, robots_filter: ObjectsFilterABC):
        super().__init__(roco_dataset.name + "_AIR")
        self.robots_filter = robots_filter
        self.roco_dataset = roco_dataset

    def __iter__(self) -> Iterator[Tuple[Image, ROCOAnnotation, str]]:
        for image, annotation, name in self.roco_dataset:
            yield from self._generate_from_single(image, annotation, name)

    def _generate_from_single(
        self, image: Image, annotation: ROCOAnnotation, name: str
    ) -> Iterator[Tuple[Image, ROCOAnnotation, str]]:
        annotation.objects, robots = ARMORS_FILTER.split(annotation.objects)
        for i, robot in enumerate(self.robots_filter.filter(robots)):
            yield crop_image_annotation(
                image, annotation, robot.box, min_coverage=0.75, name=f"{name}-{i}-{robot.type.name.lower()}"
            )


class AIRDatasetCache(DatasetCache[ROCOAnnotation]):
    def _save_one(self, img: Image, annotation: ROCOAnnotation, name: str):
        annotation.save_with_image(self.cache_dir, img, name)
