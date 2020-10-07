from unittest import TestCase

from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


class TestDJIDataset(TestCase):
    def test_north(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.DJI.NORTH_CHINA, 2474)

    def test_south(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.DJI.SOUTH_CHINA, 2555)

    def test_central(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.DJI.CENTRAL_CHINA, 2655)

    def test_final(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.DJI.FINAL, 2685)

    def assert_dataset_has_size(self, dataset_builder: ROCODatasetBuilder, size: int):
        dataset = dataset_builder.build_lazy()
        self.assertEqual(size, len(dataset))
