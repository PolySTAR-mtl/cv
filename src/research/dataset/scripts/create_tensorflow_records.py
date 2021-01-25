from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.dataset.tensorflow_record import TensorflowRecordFactory


def create_one_record_per_roco_dset():
    for datasets in ROCODatasetsZoo:
        for dataset in datasets:
            TensorflowRecordFactory.from_dataset(dataset)


if __name__ == "__main__":
    TensorflowRecordFactory.from_datasets(ROCODatasetsZoo.DEFAULT_TEST_DATASETS, "Twitch_Test_")
    TensorflowRecordFactory.from_datasets(ROCODatasetsZoo.TWITCH_TRAIN_DATASETS, "Twitch_Train_")
    TensorflowRecordFactory.from_datasets(ROCODatasetsZoo.DJI_TRAIN_DATASETS, "DJI_Train_")
    TensorflowRecordFactory.from_datasets(ROCODatasetsZoo.DJI_ZOOMED_TRAIN_DATASETS, "DJIZoomedV2_Train_")
    TensorflowRecordFactory.from_datasets(
        ROCODatasetsZoo.TWITCH_TRAIN_DATASETS + ROCODatasetsZoo.DJI_TRAIN_DATASETS, "Twitch_DJI_Train_"
    )
    TensorflowRecordFactory.from_datasets(
        ROCODatasetsZoo.TWITCH_TRAIN_DATASETS + ROCODatasetsZoo.DJI_ZOOMED_TRAIN_DATASETS, "Twitch_DJIZoomedV2_Train_"
    )
