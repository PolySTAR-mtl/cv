from itertools import islice
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from polystar.common.filters.keep_filter import KeepFilter
from polystar.common.models.object import Armor
from research.common.datasets.dataset import Dataset
from research.common.datasets.filtered_dataset import FilteredTargetsDataset
from research.common.datasets.image_dataset import open_file_dataset
from research.common.datasets.roco.zoo.roco_datasets_zoo import ROCODatasetsZoo
from research.robots_at_robots.dataset.armor_value_dataset import (
    ArmorValueDatasetCache, ArmorValueDirectoryDataset)


class ArmorDigitDirectoryDataset(ArmorValueDirectoryDataset[int]):
    @staticmethod
    def _value_from_str(label: str) -> int:
        return int(label)


class ArmorDigitDatasetCache(ArmorValueDatasetCache[str]):
    def __init__(self, acceptable_digits: Iterable[int]):
        super().__init__("digits")
        self.acceptable_digits = acceptable_digits

    def _value_from_armor(self, armor: Armor) -> int:
        return armor.number

    def from_directory_and_name(self, directory: Path, name: str) -> Dataset[Path, int]:
        full_dataset = ArmorDigitDirectoryDataset(directory, name)
        return FilteredTargetsDataset(full_dataset, KeepFilter(self.acceptable_digits))


if __name__ == "__main__":
    _dataset = open_file_dataset(
        ArmorDigitDatasetCache((1, 2, 3, 4, 5, 7)).from_roco_dataset(ROCODatasetsZoo.TWITCH.T470150052)
    )

    for _image, _value, _name in islice(_dataset, 40, 50):
        print(_value)
        plt.imshow(_image)
        plt.show()
        plt.clf()
