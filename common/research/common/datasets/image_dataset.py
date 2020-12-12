from pathlib import Path

from polystar.common.models.image import Image
from research.common.datasets.dataset import Dataset
from research.common.datasets.lazy_dataset import LazyDataset, TargetT

LazyFileDataset = LazyDataset[Path, TargetT]
FileDataset = Dataset[Path, TargetT]

LazyImageDataset = LazyDataset[Image, TargetT]
ImageDataset = Dataset[Image, TargetT]
