from research.common.constants import DJI_ROCO_DSET_DIR
from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.common.datasets.roco.roco_datasets import ROCODatasets


class DJIROCODatasets(ROCODatasets):
    CentralChina = "robomaster_Central China Regional Competition"
    NorthChina = "robomaster_North China Regional Competition"
    SouthChina = "robomaster_South China Regional Competition"
    Final = "robomaster_Final Tournament"

    @staticmethod
    def make_dataset(dataset_name: str, competition_name: str) -> DirectoryROCODataset:
        return DirectoryROCODataset(DJI_ROCO_DSET_DIR / competition_name, dataset_name)
