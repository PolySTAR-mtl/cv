from itertools import islice
from typing import List, Set, Tuple

from polystar.filters.exclude_filter import ExcludeFilter
from polystar.models.image import FileImage
from polystar.models.roco_object import Armor, ArmorDigit
from research.common.datasets.dataset import Dataset
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots.dataset.armor_value_dataset_generator import ArmorValueDatasetGenerator
from research.robots.dataset.armor_value_target_factory import ArmorValueTargetFactory

VALID_NUMBERS_2021: Set[int] = {1, 3, 4}  # University League


def default_armor_digit_datasets() -> Tuple[
    List[Dataset[FileImage, ArmorDigit]], List[Dataset[FileImage, ArmorDigit]], List[Dataset[FileImage, ArmorDigit]]
]:
    return make_armor_digit_dataset_generator().default_datasets()


def make_armor_digit_dataset_generator() -> ArmorValueDatasetGenerator[ArmorDigit]:
    return ArmorValueDatasetGenerator("digits", ArmorDigitTargetFactory(), ExcludeFilter({ArmorDigit.OUTDATED}))


class ArmorDigitTargetFactory(ArmorValueTargetFactory[ArmorDigit]):
    def from_str(self, label: str) -> ArmorDigit:
        n = int(label)

        if n in VALID_NUMBERS_2021:  # CHANGING
            return ArmorDigit(n - (n >= 3))  # hacky, but digit 2 is absent

        return ArmorDigit.OUTDATED

    def from_armor(self, armor: Armor) -> ArmorDigit:
        return ArmorDigit(armor.number) if armor.number else ArmorDigit.UNKNOWN


if __name__ == "__main__":
    _roco_dataset_builder = ROCODatasetsZoo.DJI.CENTRAL_CHINA
    _armor_digit_dataset = make_armor_digit_dataset_generator().from_roco_dataset(_roco_dataset_builder)

    for p, c, _name in islice(_armor_digit_dataset, 20, 30):
        print(p, c, _name)
