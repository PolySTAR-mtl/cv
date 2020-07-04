from pathlib import Path
from typing import Set

from polystar.common.models.object import Armor
from research.robots_at_robots.dataset.armor_image_dataset_factory import ArmorImageDatasetGenerator


class ArmorDigitDatasetGenerator(ArmorImageDatasetGenerator[int]):
    task_name: str = "digits"

    def __init__(self, acceptable_digits: Set[int]):
        self.acceptable_digits = acceptable_digits

    def _label_from_str(self, label: str) -> int:
        return int(label)

    def _label_from_armor_info(self, armor: Armor, k: int, path: Path) -> int:
        return armor.number

    def _valid_label(self, label: int) -> bool:
        return label in self.acceptable_digits
