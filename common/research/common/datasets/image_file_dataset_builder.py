from pathlib import Path
from typing import Callable, Generic, Iterable, Iterator, Tuple

from polystar.common.models.image import FileImage, Image, load_image
from research.common.datasets.dataset_builder import DatasetBuilder
from research.common.datasets.lazy_dataset import LazyDataset, TargetT


class LazyFileDataset(LazyDataset[Path, TargetT]):
    def __init__(self, files: Iterable[Path], target_from_file: Callable[[Path], TargetT], name: str):
        super().__init__(name)
        self.target_from_file = target_from_file
        self.files = sorted(files)

    def __iter__(self) -> Iterator[Tuple[Path, TargetT, str]]:
        for file in self.files:
            yield file, self.target_from_file(file), file.stem

    def __len__(self):
        return len(self.files)


class FileDatasetBuilder(Generic[TargetT], DatasetBuilder[Path, TargetT]):
    def __init__(self, dataset: LazyFileDataset):
        super().__init__(dataset)

    def to_images(self) -> DatasetBuilder[Image, TargetT]:
        return self.transform_examples(load_image)

    def to_file_images(self) -> DatasetBuilder[FileImage, TargetT]:
        return self.transform_examples(FileImage)


class DirectoryDatasetBuilder(FileDatasetBuilder[TargetT]):
    def __init__(self, directory: Path, target_from_file: Callable[[Path], TargetT], name: str, extension: str = "jpg"):
        super().__init__(LazyFileDataset(directory.glob(f"*.{extension}"), target_from_file, name))
        self.images_dir = directory
