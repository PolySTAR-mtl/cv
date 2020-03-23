from typing import Tuple, Sequence

from polystar.common.models.image import Image
from research.dataset.armor_dataset_factory import ArmorDatasetFactory
from research_common.dataset.roco_dataset import ROCODataset
from research_common.image_pipeline_evaluation.image_dataset_generator import ImageDatasetGenerator


class ArmorColorDatasetGenerator(ImageDatasetGenerator[str]):
    def from_roco_dataset(self, dataset: ROCODataset) -> Tuple[Sequence[Image], Sequence[str]]:
        return zip(*[(armor_img, c.name) for (armor_img, c, n, k, p) in ArmorDatasetFactory.from_dataset(dataset)])
