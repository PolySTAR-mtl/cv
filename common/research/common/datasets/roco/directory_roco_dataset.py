from pathlib import Path
from typing import Iterable, Tuple

from polystar.common.models.image import Image
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset import ROCODataset


class DirectoryROCODataset(ROCODataset):
    def __init__(self, dataset_path: Path, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    @property
    def images_dir_path(self) -> Path:
        return self.dataset_path / "image"

    @property
    def annotation_paths(self) -> Iterable[Path]:
        return sorted(self.annotations_dir_path.glob("*.xml"))

    @property
    def annotations_dir_path(self) -> Path:
        return self.dataset_path / "image_annotation"

    def __iter__(self) -> Iterable[Tuple[Image, ROCOAnnotation]]:
        for annotation_file in self.annotation_paths:
            yield self._load_from_annotation_file(annotation_file)

    def _load_from_annotation_file(self, annotation_file: Path) -> Tuple[Image, ROCOAnnotation]:
        return (
            Image.from_path(self.images_dir_path / f"{annotation_file.stem}.jpg"),
            ROCOAnnotation.from_xml_file(annotation_file),
        )
