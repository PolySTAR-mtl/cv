from typing import List

from research.common.constants import TENSORFLOW_RECORDS_DIR
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.dataset.tensorflow_record import ROCOTensorflowRecordFactory
from research.roco_detection.robots_dataset import AnnotationHasObjectsFilter


class AirTensorflowRecordFactory(ROCOTensorflowRecordFactory):
    RECORDS_DIR = TENSORFLOW_RECORDS_DIR / "air"

    def from_builders(self, builders: List[ROCODatasetBuilder], prefix: str = ""):
        super().from_builders((b.to_air().filter_targets(AnnotationHasObjectsFilter()) for b in builders), prefix)


if __name__ == "__main__":
    _factory = AirTensorflowRecordFactory()
    _factory.from_builders(ROCODatasetsZoo.TWITCH_TRAIN_DATASETS, "Twitch2_Train")
    _factory.from_builders(ROCODatasetsZoo.TWITCH_VALIDATION_DATASETS, "Twitch2_Val")
    _factory.from_builders(ROCODatasetsZoo.TWITCH_TEST_DATASETS, "Twitch2_Test")
    _factory.from_builders(ROCODatasetsZoo.DJI, "Dji")
