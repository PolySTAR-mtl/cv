from polystar.common.utils.str_utils import camel2snake
from research.common.constants import DJI_ROCO_ZOOMED_DSET_DIR
from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.common.datasets.roco.roco_datasets import ROCODatasets


class DJIROCOZoomedDatasets(ROCODatasets):
    CentralChina = ()
    NorthChina = ()
    SouthChina = ()
    Final = ()

    @staticmethod
    def _make_dataset(dataset_name: str) -> DirectoryROCODataset:
        return DirectoryROCODataset(DJI_ROCO_ZOOMED_DSET_DIR / camel2snake(dataset_name), f"{dataset_name}ZoomedV2")
