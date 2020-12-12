from research.common.dataset.cleaning.dataset_cleaner_app import DatasetCleanerApp
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.armor_digit.armor_digit_dataset import make_armor_digit_dataset_generator

if __name__ == "__main__":
    # _roco_dataset = ROCODatasetsZoo.TWITCH.T470149568
    # _roco_dataset = ROCODatasetsZoo.TWITCH.T470150052
    # _roco_dataset = ROCODatasetsZoo.TWITCH.T470151286
    # _roco_dataset = ROCODatasetsZoo.TWITCH.T470152289
    # _roco_dataset = ROCODatasetsZoo.TWITCH.T470152730
    # _roco_dataset = ROCODatasetsZoo.TWITCH.T470152838
    # _roco_dataset = ROCODatasetsZoo.TWITCH.T470153081
    # _roco_dataset = ROCODatasetsZoo.TWITCH.T470158483

    # _roco_dataset = ROCODatasetsZoo.DJI.NORTH_CHINA # .skip(200 + (13558 - 2974))
    _roco_dataset = ROCODatasetsZoo.DJI.FINAL

    _armor_digit_dataset = (
        make_armor_digit_dataset_generator()
        .from_roco_dataset(_roco_dataset)
        .skip((1009 - 117) + (1000 - 86) + (1000 - 121) + (1000 - 138) + (1000 - 137))
        .cap(1000)
    )

    DatasetCleanerApp(_armor_digit_dataset, invalidate_key="u", validate_key="h").run()
