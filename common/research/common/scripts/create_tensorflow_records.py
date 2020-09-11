from itertools import chain

from research.common.dataset.tensorflow_record import TensorflowRecordFactory
from research.common.datasets.roco.zoo.dji import DJIROCODatasets
from research.common.datasets.roco.zoo.dji_zoomed import DJIROCOZoomedDatasets
from research.common.datasets.roco.zoo.roco_datasets_zoo import ROCODatasetsZoo
from research.common.datasets.roco.zoo.twitch import TwitchROCODatasets


def create_one_record_per_roco_dset():
    for roco_set in chain(*(datasets for datasets in ROCODatasetsZoo())):
        TensorflowRecordFactory.from_dataset(roco_set)


def create_twitch_records():
    TensorflowRecordFactory.from_datasets(
        [
            TwitchROCODatasets.TWITCH_470149568,
            TwitchROCODatasets.TWITCH_470150052,
            TwitchROCODatasets.TWITCH_470151286,
            TwitchROCODatasets.TWITCH_470152289,
            TwitchROCODatasets.TWITCH_470152730,
        ],
        "Twitch_Train_",
    )
    TensorflowRecordFactory.from_datasets(
        [TwitchROCODatasets.TWITCH_470152838, TwitchROCODatasets.TWITCH_470153081, TwitchROCODatasets.TWITCH_470158483],
        "Twitch_Test_",
    )


def create_dji_records():
    TensorflowRecordFactory.from_datasets(
        [DJIROCODatasets.CentralChina, DJIROCODatasets.NorthChina, DJIROCODatasets.SouthChina], "DJI_Train_"
    )
    TensorflowRecordFactory.from_dataset(DJIROCODatasets.Final, "DJI_Test_")


def create_dji_zoomed_records():
    TensorflowRecordFactory.from_datasets(
        [DJIROCOZoomedDatasets.CentralChina, DJIROCOZoomedDatasets.NorthChina, DJIROCOZoomedDatasets.SouthChina],
        "DJIZoomedV2_Train_",
    )
    TensorflowRecordFactory.from_dataset(DJIROCOZoomedDatasets.Final, "DJIZoomedV2_Test_")


if __name__ == "__main__":
    # create_one_record_per_roco_dset()
    create_twitch_records()
    create_dji_records()
    create_dji_zoomed_records()
