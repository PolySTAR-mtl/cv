from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from polystar.common.models.image import Image
from polystar.common.models.image_annotation import ImageAnnotation


@dataclass
class Dataset:

    dataset_path: Path
    dataset_name: str

    @property
    def images_dir_path(self) -> Path:
        return self.dataset_path / "image"

    @property
    def annotations_dir_path(self) -> Path:
        return self.dataset_path / "image_annotation"

    @property
    def image_paths(self) -> Iterable[Path]:
        return self.images_dir_path.glob("*.jpg")

    @property
    def annotation_paths(self) -> Iterable[Path]:
        return self.annotations_dir_path.glob("*.xml")

    @property
    def images(self) -> Iterable[Image]:
        for image_path in self.image_paths:
            yield Image.from_path(image_path)

    @property
    def image_annotations(self) -> Iterable[ImageAnnotation]:
        for annotation_path in self.annotation_paths:
            yield ImageAnnotation.from_xml_file(annotation_path)
