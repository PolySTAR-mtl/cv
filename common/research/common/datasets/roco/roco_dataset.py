from polystar.common.models.image import Image
from research.common.datasets.dataset import Dataset
from research.common.datasets.image_dataset import ImageFileDataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation

ROCODataset = Dataset[Image, ROCOAnnotation]
ROCOFileDataset = ImageFileDataset[ROCOAnnotation]
