from itertools import islice
from typing import Iterable

from polystar.common.filters.keep_filter import KeepFilter
from polystar.common.models.object import Armor
from research.common.datasets_v3.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.dataset.armor_value_dataset_generator import ArmorValueDatasetGenerator
from research.robots_at_robots.dataset.armor_value_target_factory import ArmorValueTargetFactory


class ArmorDigitTargetFactory(ArmorValueTargetFactory[int]):
    def from_str(self, label: str) -> int:
        return int(label)

    def from_armor(self, armor: Armor) -> int:
        return armor.number


def make_armor_digit_dataset_generator(acceptable_digits: Iterable[int]) -> ArmorValueDatasetGenerator[int]:
    return ArmorValueDatasetGenerator("digits", ArmorDigitTargetFactory(), KeepFilter(set(acceptable_digits)))


if __name__ == "__main__":
    _roco_dataset_builder = ROCODatasetsZoo.DJI.CENTRAL_CHINA.builder
    _armor_digit_dataset = make_armor_digit_dataset_generator([1, 2]).from_roco_dataset(_roco_dataset_builder)

    for p, c, _name in islice(_armor_digit_dataset, 20, 30):
        print(p, c, _name)
