from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
from polystar.common.models.object import Armor
from research.common.datasets.dataset import Dataset
from research.common.datasets.image_dataset import open_file_dataset
from research.common.datasets.roco.zoo.roco_datasets_zoo import ROCODatasetsZoo
from research.robots_at_robots.dataset.armor_value_dataset import (
    ArmorValueDatasetCache, ArmorValueDirectoryDataset)


class ArmorColorDirectoryDataset(ArmorValueDirectoryDataset[str]):
    @staticmethod
    def _value_from_str(label: str) -> str:
        return label


class ArmorColorDatasetCache(ArmorValueDatasetCache[str]):
    def __init__(self):
        super().__init__("colors")

    def _value_from_armor(self, armor: Armor) -> str:
        return armor.color.name.lower()

    def from_directory_and_name(self, directory: Path, name: str) -> Dataset[Path, str]:
        return ArmorColorDirectoryDataset(directory, name)


if __name__ == "__main__":
    _dataset = open_file_dataset(ArmorColorDatasetCache().from_roco_dataset(ROCODatasetsZoo.TWITCH.T470150052))

    for _image, _value, _name in islice(_dataset, 40, 50):
        print(_value)
        plt.imshow(_image)
        plt.show()
        plt.clf()
