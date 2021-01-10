from itertools import islice
from typing import List, Set, Tuple

from polystar.common.filters.exclude_filter import ExcludeFilter
from polystar.common.models.object import Armor, ArmorDigit
from research.common.datasets.image_dataset import FileImageDataset
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots.dataset.armor_value_dataset_generator import ArmorValueDatasetGenerator
from research.robots.dataset.armor_value_target_factory import ArmorValueTargetFactory

VALID_NUMBERS_2021: Set[int] = {1, 3, 4}  # University League


def make_armor_digit_dataset_generator() -> ArmorValueDatasetGenerator[ArmorDigit]:
    return ArmorValueDatasetGenerator("digits", ArmorDigitTargetFactory(), ExcludeFilter({ArmorDigit.OUTDATED}))


def default_armor_digit_datasets() -> Tuple[List[FileImageDataset], List[FileImageDataset], List[FileImageDataset]]:
    digit_dataset_generator = make_armor_digit_dataset_generator()
    train_datasets, validation_datasets, test_datasets = digit_dataset_generator.from_roco_datasets(
        [
            ROCODatasetsZoo.TWITCH.T470150052,
            ROCODatasetsZoo.TWITCH.T470152730,
            ROCODatasetsZoo.TWITCH.T470153081,
            ROCODatasetsZoo.TWITCH.T470158483,
            ROCODatasetsZoo.DJI.FINAL,
            ROCODatasetsZoo.DJI.CENTRAL_CHINA,
            ROCODatasetsZoo.DJI.NORTH_CHINA,
            ROCODatasetsZoo.DJI.SOUTH_CHINA,
        ],
        [ROCODatasetsZoo.TWITCH.T470149568, ROCODatasetsZoo.TWITCH.T470152289],
        [ROCODatasetsZoo.TWITCH.T470152838, ROCODatasetsZoo.TWITCH.T470151286],
    )
    # train_datasets.append(
    #     digit_dataset_generator.from_roco_dataset(ROCODatasetsZoo.DJI.FINAL).to_file_images()
    #     # .cap(2133 + 1764 + 1436)
    #     # .skip(2133 + 176 + 1436)
    #     # .cap(5_000)
    #     .build()
    # )
    return train_datasets, validation_datasets, test_datasets


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
