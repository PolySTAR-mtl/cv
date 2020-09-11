from pathlib import Path
from typing import Iterable, List, Tuple

from more_itertools import ilen
from polystar.common.models.image import Image
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset import ROCODataset


class DirectoryROCODataset(ROCODataset):
    def __init__(self, dataset_path: Path, name: str):
        super().__init__(name)
        self.dataset_path = dataset_path

    @property
    def targets(self) -> List[ROCOAnnotation]:
        if self._is_loaded:
            return super().targets
        return list(map(ROCOAnnotation.from_xml_file, self.annotation_paths))

    @property
    def images_dir_path(self) -> Path:
        return self.dataset_path / "image"

    @property
    def annotation_paths(self) -> Iterable[Path]:
        return sorted(self.annotations_dir_path.glob("*.xml"))

    @property
    def annotations_dir_path(self) -> Path:
        return self.dataset_path / "image_annotation"

    def __len__(self):
        return ilen(self.annotation_paths)

    def __iter__(self) -> Iterable[Tuple[Image, ROCOAnnotation]]:
        for annotation_file in self.annotation_paths:
            yield self._load_from_annotation_file(annotation_file)

    def _load_from_annotation_file(self, annotation_file: Path) -> Tuple[Image, ROCOAnnotation]:
        return (
            Image.from_path(self.images_dir_path / f"{annotation_file.stem}.jpg"),
            ROCOAnnotation.from_xml_file(annotation_file),
        )

    def save_one(self, image: Image, annotation: ROCOAnnotation):
        Image.save(image, self.images_dir_path / f"{annotation.name}.jpg")
        (self.annotations_dir_path / f"{annotation.name}.xml").write_text(annotation.to_xml())
