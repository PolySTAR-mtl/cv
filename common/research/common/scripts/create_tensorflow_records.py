from research.common.dataset.tensorflow_record import TensorflowRecordFactory
from research.common.datasets_v3.roco.zoo.dji import DJIROCODatasets
from research.common.datasets_v3.roco.zoo.dji_zoomed import DJIROCOZoomedDatasets
from research.common.datasets_v3.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.common.datasets_v3.roco.zoo.twitch import TwitchROCODatasets


def create_one_record_per_roco_dset():
    for datasets in ROCODatasetsZoo:
        for dataset in datasets:
            TensorflowRecordFactory.from_dataset(dataset)


def create_twitch_records():
    TensorflowRecordFactory.from_datasets(
        [
            TwitchROCODatasets.T470149568,
            TwitchROCODatasets.T470150052,
            TwitchROCODatasets.T470151286,
            TwitchROCODatasets.T470152289,
            TwitchROCODatasets.T470152730,
        ],
        "Twitch_Train_",
    )
    TensorflowRecordFactory.from_datasets(
        [TwitchROCODatasets.T470152838, TwitchROCODatasets.T470153081, TwitchROCODatasets.T470158483], "Twitch_Test_",
    )


def create_dji_records():
    TensorflowRecordFactory.from_datasets(
        [DJIROCODatasets.CENTRAL_CHINA, DJIROCODatasets.NORTH_CHINA, DJIROCODatasets.SOUTH_CHINA], "DJI_Train_"
    )
    TensorflowRecordFactory.from_dataset(DJIROCODatasets.FINAL, "DJI_Test_")


def create_dji_zoomed_records():
    TensorflowRecordFactory.from_datasets(
        [DJIROCOZoomedDatasets.CENTRAL_CHINA, DJIROCOZoomedDatasets.NORTH_CHINA, DJIROCOZoomedDatasets.SOUTH_CHINA],
        "DJIZoomedV2_Train_",
    )
    TensorflowRecordFactory.from_dataset(DJIROCOZoomedDatasets.FINAL, "DJIZoomedV2_Test_")


if __name__ == "__main__":
    # create_one_record_per_roco_dset()
    create_twitch_records()
    create_dji_records()
    create_dji_zoomed_records()
