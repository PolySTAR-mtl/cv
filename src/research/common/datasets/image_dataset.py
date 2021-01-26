from pathlib import Path

from polystar.models.image import FileImage, Image
from research.common.datasets.dataset import Dataset
from research.common.datasets.lazy_dataset import LazyDataset, TargetT

LazyFileDataset = LazyDataset[Path, TargetT]
FileDataset = Dataset[Path, TargetT]

LazyImageDataset = LazyDataset[Image, TargetT]
ImageDataset = Dataset[Image, TargetT]

LazyFileImageDataset = LazyDataset[FileImage, TargetT]
FileImageDataset = Dataset[FileImage, TargetT]
