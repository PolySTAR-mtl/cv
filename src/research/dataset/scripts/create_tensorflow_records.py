from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.dataset.tensorflow_record import TensorflowRecordFactory

if __name__ == "__main__":
    _factory = TensorflowRecordFactory()
    # _factory.from_datasets(ROCODatasetsZoo.TWITCH_TRAIN_DATASETS, "Twitch2_Train_")
    # _factory.from_datasets(ROCODatasetsZoo.TWITCH_VALIDATION_DATASETS, "Twitch2_Val_")
    # _factory.from_datasets(ROCODatasetsZoo.TWITCH_TEST_DATASETS, "Twitch2_Test_")
    _factory.from_datasets(ROCODatasetsZoo.TWITCH_DJI_TRAIN_DATASETS, "Twitch2_Dji_Train_")
    # _factory.from_datasets(ROCODatasetsZoo.TWITCH_DJI_ZOOMED_TRAIN_DATASETS, "Twitch2_DjiZoomed2_Train_")

    # TensorflowRecordFactory.from_datasets(ROCODatasetsZoo.DJI_TRAIN_DATASETS, "DJI_Train_")
    # TensorflowRecordFactory.from_datasets(ROCODatasetsZoo.DJI_ZOOMED_TRAIN_DATASETS, "DJIZoomedV2_Train_")
    # TensorflowRecordFactory.from_datasets(
    #     ROCODatasetsZoo.TWITCH_TRAIN_DATASETS + ROCODatasetsZoo.DJI_TRAIN_DATASETS, "Twitch_DJI_Train_"
    # )
    # TensorflowRecordFactory.from_datasets(
    #     ROCODatasetsZoo.TWITCH_TRAIN_DATASETS + ROCODatasetsZoo.DJI_ZOOMED_TRAIN_DATASETS, "Twitch_DJIZoomedV2_Train_"
    # )
