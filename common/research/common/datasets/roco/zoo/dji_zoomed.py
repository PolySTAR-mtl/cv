from typing import Any

from polystar.common.utils.str_utils import camel2snake
from research.common.constants import DJI_ROCO_ZOOMED_DSET_DIR
from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.common.datasets.roco.roco_datasets import ROCODatasets


class DJIROCOZoomedDatasets(ROCODatasets):
    directory = DJI_ROCO_ZOOMED_DSET_DIR

    CentralChina = ()
    NorthChina = ()
    SouthChina = ()
    Final = ()

    @classmethod
    def make_dataset(cls, dataset_name: str, *args: Any) -> DirectoryROCODataset:
        return DirectoryROCODataset(cls.directory / camel2snake(dataset_name), f"{dataset_name}ZoomedV2")
