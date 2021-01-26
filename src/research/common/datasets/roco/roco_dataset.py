from pathlib import Path

from polystar.models.image import Image
from research.common.datasets.dataset import Dataset
from research.common.datasets.lazy_dataset import LazyDataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation

LazyROCOFileDataset = LazyDataset[Path, ROCOAnnotation]
ROCOFileDataset = Dataset[Path, ROCOAnnotation]

LazyROCODataset = LazyDataset[Image, ROCOAnnotation]
ROCODataset = Dataset[Image, ROCOAnnotation]
