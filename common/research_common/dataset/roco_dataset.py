from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from polystar.common.models.image import Image
from polystar.common.models.image_annotation import ImageAnnotation


@dataclass
class ROCODataset:
    image_paths: Iterable[Path]
    annotation_paths: Iterable[Path]
    dataset_name: str

    @property
    def images(self) -> Iterable[Image]:
        for image_path in self.image_paths:
            yield Image.from_path(image_path)

    @property
    def image_annotations(self) -> Iterable[ImageAnnotation]:
        for annotation_path in self.annotation_paths:
            yield ImageAnnotation.from_xml_file(annotation_path)
