from typing import List, Tuple

from polystar.common.models.object import Armor, ArmorColor
from research.common.datasets.image_dataset import FileImageDataset
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots.dataset.armor_value_dataset_generator import ArmorValueDatasetGenerator
from research.robots.dataset.armor_value_target_factory import ArmorValueTargetFactory


class ArmorColorTargetFactory(ArmorValueTargetFactory[ArmorColor]):
    def from_str(self, label: str) -> ArmorColor:
        return ArmorColor(label)

    def from_armor(self, armor: Armor) -> ArmorColor:
        return armor.color


def make_armor_color_dataset_generator() -> ArmorValueDatasetGenerator[ArmorColor]:
    return ArmorValueDatasetGenerator("colors", ArmorColorTargetFactory())


def make_armor_color_datasets(
    include_dji: bool = True,
) -> Tuple[List[FileImageDataset], List[FileImageDataset], List[FileImageDataset]]:
    color_dataset_generator = make_armor_color_dataset_generator()

    train_roco_datasets = [
        ROCODatasetsZoo.TWITCH.T470150052,
        ROCODatasetsZoo.TWITCH.T470152730,
        ROCODatasetsZoo.TWITCH.T470153081,
        ROCODatasetsZoo.TWITCH.T470158483,
    ]
    if include_dji:
        train_roco_datasets.extend(
            [
                ROCODatasetsZoo.DJI.FINAL,
                ROCODatasetsZoo.DJI.CENTRAL_CHINA,
                ROCODatasetsZoo.DJI.NORTH_CHINA,
                ROCODatasetsZoo.DJI.SOUTH_CHINA,
            ]
        )

    train_datasets, validation_datasets, test_datasets = color_dataset_generator.from_roco_datasets(
        train_roco_datasets,
        [ROCODatasetsZoo.TWITCH.T470149568, ROCODatasetsZoo.TWITCH.T470152289],
        [ROCODatasetsZoo.TWITCH.T470152838, ROCODatasetsZoo.TWITCH.T470151286],
    )

    return train_datasets, validation_datasets, test_datasets
