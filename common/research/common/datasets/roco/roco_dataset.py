from research.common.datasets.image_dataset import ImageDataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation

ROCODataset = ImageDataset[ROCOAnnotation]
