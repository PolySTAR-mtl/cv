from pathlib import Path

from polystar.common.models.image import Image
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets_v3.dataset import Dataset
from research.common.datasets_v3.lazy_dataset import LazyDataset

LazyROCOFileDataset = LazyDataset[Path, ROCOAnnotation]
ROCOFileDataset = Dataset[Path, ROCOAnnotation]

LazyROCODataset = LazyDataset[Image, ROCOAnnotation]
ROCODataset = Dataset[Image, ROCOAnnotation]
