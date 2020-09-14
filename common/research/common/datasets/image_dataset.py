from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Tuple

from memoized_property import memoized_property
from more_itertools import ilen
from polystar.common.models.image import Image, load_image
from research.common.datasets.dataset import Dataset, LazyDataset, TargetT

ImageDataset = Dataset[Image, TargetT]


class ImageFileDataset(LazyDataset[Path, TargetT], ABC):
    def __iter__(self) -> Iterator[Tuple[Path, TargetT, str]]:
        for image_file in self.image_files:
            yield image_file, self.target_from_image_file(image_file), image_file.stem

    @abstractmethod
    def target_from_image_file(self, image_file: Path) -> TargetT:
        pass

    @property
    @abstractmethod
    def image_files(self) -> Iterator[Path]:
        pass

    def open(self) -> ImageDataset:
        return open_file_dataset(self)

    def __len__(self):
        return ilen(self.image_files)


def open_file_dataset(dataset: Dataset[Path, TargetT]) -> ImageDataset:
    return dataset.transform_examples(load_image)


class ImageDirectoryDataset(ImageFileDataset[TargetT], ABC):
    def __init__(self, images_dir: Path, name: str, extension: str = "jpg"):
        super().__init__(name)
        self.extension = extension
        self.images_dir = images_dir

    @memoized_property
    def image_files(self) -> List[Path]:
        return list(sorted(self.images_dir.glob(f"*.{self.extension}")))

    def __len__(self):
        return len(self.image_files)
