from typing import List, Tuple

from polystar.models.image import FileImage
from polystar.models.roco_object import Armor, ArmorColor
from research.common.datasets.dataset import Dataset
from research.robots.dataset.armor_value_dataset_generator import ArmorValueDatasetGenerator
from research.robots.dataset.armor_value_target_factory import ArmorValueTargetFactory


def make_armor_color_datasets(
    include_dji: bool = True,
) -> Tuple[
    List[Dataset[FileImage, ArmorColor]], List[Dataset[FileImage, ArmorColor]], List[Dataset[FileImage, ArmorColor]]
]:
    return make_armor_color_dataset_generator().default_datasets(include_dji)


class ArmorColorTargetFactory(ArmorValueTargetFactory[ArmorColor]):
    def from_str(self, label: str) -> ArmorColor:
        return ArmorColor(label)

    def from_armor(self, armor: Armor) -> ArmorColor:
        return armor.color


def make_armor_color_dataset_generator() -> ArmorValueDatasetGenerator[ArmorColor]:
    return ArmorValueDatasetGenerator("colors", ArmorColorTargetFactory())
