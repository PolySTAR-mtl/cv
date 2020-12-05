from itertools import islice

from polystar.common.models.object import Armor, ArmorColor
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.dataset.armor_value_dataset_generator import ArmorValueDatasetGenerator
from research.robots_at_robots.dataset.armor_value_target_factory import ArmorValueTargetFactory


class ArmorColorTargetFactory(ArmorValueTargetFactory[ArmorColor]):
    def from_str(self, label: str) -> ArmorColor:
        return ArmorColor(label)

    def from_armor(self, armor: Armor) -> ArmorColor:
        return armor.color


def make_armor_color_dataset_generator() -> ArmorValueDatasetGenerator[ArmorColor]:
    return ArmorValueDatasetGenerator("colors", ArmorColorTargetFactory())


if __name__ == "__main__":
    _roco_dataset_builder = ROCODatasetsZoo.DJI.CENTRAL_CHINA
    _armor_color_dataset = make_armor_color_dataset_generator().from_roco_dataset(_roco_dataset_builder)

    for p, c, _name in islice(_armor_color_dataset, 20, 25):
        print(p, c, _name)
