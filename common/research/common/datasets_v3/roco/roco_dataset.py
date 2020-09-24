from pathlib import Path

from polystar.common.models.image import Image
from research.common.datasets_v3.dataset import Dataset
from research.common.datasets_v3.lazy_dataset import LazyDataset
from research.common.datasets_v3.roco.roco_annotation import ROCOAnnotation

LazyROCOFileDataset = LazyDataset[Path, ROCOAnnotation]
ROCOFileDataset = Dataset[Path, ROCOAnnotation]

LazyROCODataset = LazyDataset[Image, ROCOAnnotation]
ROCODataset = Dataset[Image, ROCOAnnotation]
