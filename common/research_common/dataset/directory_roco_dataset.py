from pathlib import Path
from typing import Iterable

from research_common.dataset.roco_dataset import ROCODataset


class DirectoryROCODataset(ROCODataset):
    def __init__(self, dataset_path: Path, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

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
