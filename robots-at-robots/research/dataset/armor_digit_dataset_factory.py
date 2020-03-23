from typing import Tuple, Sequence, Set

from polystar.common.models.image import Image
from research.dataset.armor_dataset_factory import ArmorDatasetFactory
from research_common.dataset.roco_dataset import ROCODataset
from research_common.image_pipeline_evaluation.image_dataset_generator import ImageDatasetGenerator


class ArmorDigitDatasetGenerator(ImageDatasetGenerator[str]):
    def __init__(self, acceptable_digits: Set[int]):
        self.acceptable_digits = acceptable_digits

    def from_roco_dataset(self, dataset: ROCODataset) -> Tuple[Sequence[Image], Sequence[int]]:
        return zip(
            *[
                (armor_img, digit)
                for (armor_img, c, digit, k, p) in ArmorDatasetFactory.from_dataset(dataset)
                if digit in self.acceptable_digits
            ]
        )
