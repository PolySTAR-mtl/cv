from unittest import TestCase

from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


class TestDJIDataset(TestCase):
    def test_north(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.DJI_ZOOMED.NORTH_CHINA, 5474)

    def test_south(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.DJI_ZOOMED.SOUTH_CHINA, 5272)

    def test_central(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.DJI_ZOOMED.CENTRAL_CHINA, 5307)

    def test_final(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.DJI_ZOOMED.FINAL, 5260)

    def assert_dataset_has_size(self, dataset_builder: ROCODatasetBuilder, size: int):
        dataset = dataset_builder.build_lazy()
        self.assertEqual(size, len(dataset))
