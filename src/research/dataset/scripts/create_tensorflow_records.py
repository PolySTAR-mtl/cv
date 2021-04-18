from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.dataset.tensorflow_record import ROCOTensorflowRecordFactory

if __name__ == "__main__":
    _factory = ROCOTensorflowRecordFactory()
    _factory.from_builders(ROCODatasetsZoo.TWITCH_TRAIN_DATASETS, "Twitch2_Train")
    _factory.from_builders(ROCODatasetsZoo.TWITCH_VALIDATION_DATASETS, "Twitch2_Val")
    _factory.from_builders(ROCODatasetsZoo.TWITCH_TEST_DATASETS, "Twitch2_Test")
    _factory.from_builders(ROCODatasetsZoo.DJI, "Dji")
    _factory.from_builders(ROCODatasetsZoo.DJI_ZOOMED, "DjiZoomed")
