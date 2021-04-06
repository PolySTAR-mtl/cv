from pathlib import Path

from polystar.models.image import Image
from polystar.models.roco_object import Armor, ObjectType, ROCOObject
from polystar.target_pipeline.objects_filters.type_object_filter import TypeObjectsFilter
from research.common.constants import DSET_DIR
from research.common.datasets.dataset_builder import DatasetBuilder
from research.common.datasets.image_file_dataset_builder import DirectoryDatasetBuilder
from research.common.datasets.roco.air_dataset import AIRDataset, AIRDatasetCache
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_objects_dataset import ROCOObjectsDataset
from research.roco_detection.small_base_filter import SMALL_BASE_FILTER


class ROCODatasetBuilder(DirectoryDatasetBuilder[ROCOAnnotation]):
    def __init__(self, directory: Path, name: str, extension: str = "jpg"):
        super().__init__(directory / "image", self._roco_annotation_from_image_file, name, extension)
        self.annotations_dir = directory / "image_annotation"
        self.main_dir = directory

    def to_objects(self) -> DatasetBuilder[Image, ROCOObject]:
        return DatasetBuilder(ROCOObjectsDataset(self.to_images()))

    def to_armors(self) -> DatasetBuilder[Image, Armor]:
        builder = self.to_objects().filter_targets(TypeObjectsFilter({ObjectType.ARMOR}))
        builder.name = builder.name.replace("objects", "armors")
        return builder

    # FIXME: it makes no sense to have a ROCODatasetBuilder as output
    def to_air(self) -> "ROCODatasetBuilder":
        cache_dir = DSET_DIR / "air" / self.main_dir.relative_to(DSET_DIR)
        AIRDatasetCache(cache_dir, AIRDataset(self.to_images(), SMALL_BASE_FILTER)).generate_if_missing()
        return ROCODatasetBuilder(cache_dir, self.name + "_AIR")

    def _roco_annotation_from_image_file(self, image_file: Path) -> ROCOAnnotation:
        return ROCOAnnotation.from_xml_file(self.annotations_dir / f"{image_file.stem}.xml")
