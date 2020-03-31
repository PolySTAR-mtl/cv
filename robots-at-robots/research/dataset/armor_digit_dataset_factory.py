from pathlib import Path
from typing import Set

from polystar.common.models.object import ArmorColor
from research.dataset.armor_image_dataset_factory import ArmorImageDatasetGenerator


class ArmorDigitDatasetGenerator(ArmorImageDatasetGenerator[int]):
    def __init__(self, acceptable_digits: Set[int]):
        self.acceptable_digits = acceptable_digits

    def _label(self, color: ArmorColor, number: int, k: int, path: Path) -> int:
        return number

    def _valid_label(self, label: int) -> bool:
        return label in self.acceptable_digits
