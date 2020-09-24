from pathlib import Path

from polystar.common.models.image import Image
from research.common.datasets_v3.dataset import Dataset
from research.common.datasets_v3.lazy_dataset import TargetT

LazyFileDataset = Dataset[Path, TargetT]
FileDataset = Dataset[Path, TargetT]

LazyImageDataset = Dataset[Image, TargetT]
ImageDataset = Dataset[Image, TargetT]
