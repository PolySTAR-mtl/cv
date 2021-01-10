from unittest import TestCase

from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


class TestTwitchDataset(TestCase):
    def test_sizes(self):
        self.assert_dataset_has_size(ROCODatasetsZoo.TWITCH.T470149568, 372)
        self.assert_dataset_has_size(ROCODatasetsZoo.TWITCH.T470150052, 186)
        self.assert_dataset_has_size(ROCODatasetsZoo.TWITCH.T470151286, 841)
        self.assert_dataset_has_size(ROCODatasetsZoo.TWITCH.T470152289, 671)
        self.assert_dataset_has_size(ROCODatasetsZoo.TWITCH.T470152730, 367)
        self.assert_dataset_has_size(ROCODatasetsZoo.TWITCH.T470152838, 161)
        self.assert_dataset_has_size(ROCODatasetsZoo.TWITCH.T470153081, 115)
        self.assert_dataset_has_size(ROCODatasetsZoo.TWITCH.T470158483, 66)

    def assert_dataset_has_size(self, dataset_builder: ROCODatasetBuilder, size: int):
        dataset = dataset_builder.build_lazy()
        self.assertEqual(size, len(dataset))
