from pathlib import Path

from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets_v3.image_file_dataset_builder import DirectoryDatasetBuilder


class ROCODatasetBuilder(DirectoryDatasetBuilder):
    def __init__(self, directory: Path, name: str, extension: str = "jpg"):
        super().__init__(directory / "image", self.roco_annotation_from_image_file, name, extension)
        self.annotations_dir = directory / "image_annotation"
        self.main_dir = directory

    def roco_annotation_from_image_file(self, image_file: Path) -> ROCOAnnotation:
        return ROCOAnnotation.from_xml_file(self.annotations_dir / f"{image_file.stem}.xml")
