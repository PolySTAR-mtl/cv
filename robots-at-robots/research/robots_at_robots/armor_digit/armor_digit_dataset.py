from itertools import islice

from polystar.common.filters.exclude_filter import ExcludeFilter
from polystar.common.models.object import Armor, ArmorDigit
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.dataset.armor_value_dataset_generator import ArmorValueDatasetGenerator
from research.robots_at_robots.dataset.armor_value_target_factory import ArmorValueTargetFactory


class ArmorDigitTargetFactory(ArmorValueTargetFactory[ArmorDigit]):
    def from_str(self, label: str) -> ArmorDigit:
        n = int(label)

        if 1 <= n <= 5:  # CHANGING
            return ArmorType(n)

        return ArmorDigit.OUTDATED

    def from_armor(self, armor: Armor) -> ArmorDigit:
        return ArmorType(armor.number)


def make_armor_digit_dataset_generator() -> ArmorValueDatasetGenerator[int]:
    return ArmorValueDatasetGenerator("digits", ArmorDigitTargetFactory(), ExcludeFilter({ArmorDigit.OUTDATED}))


if __name__ == "__main__":
    _roco_dataset_builder = ROCODatasetsZoo.DJI.CENTRAL_CHINA
    _armor_digit_dataset = make_armor_digit_dataset_generator().from_roco_dataset(_roco_dataset_builder)

    for p, c, _name in islice(_armor_digit_dataset, 20, 30):
        print(p, c, _name)
