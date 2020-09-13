from pathlib import Path

from polystar.common.models.image import Image, save_image
from research.common.datasets.image_dataset import ImageDirectoryDataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation


class DirectoryROCODataset(ImageDirectoryDataset[ROCOAnnotation]):
    def __init__(self, dataset_path: Path, name: str):
        super().__init__(dataset_path / "image", name)
        self.main_dir = dataset_path
        self.annotations_dir: Path = self.main_dir / "image_annotation"

    def target_from_image_file(self, image_file: Path) -> ROCOAnnotation:
        return ROCOAnnotation.from_xml_file(self.annotations_dir / f"{image_file.stem}.xml")

    def create(self):
        self.main_dir.mkdir(parents=True)
        self.images_dir.mkdir()
        self.annotations_dir.mkdir()

    def add(self, image: Image, annotation: ROCOAnnotation, name: str):
        save_image(image, self.images_dir / f"{name}.jpg")
        (self.annotations_dir / f"{name}.xml").write_text(annotation.to_xml())
