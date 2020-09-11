from research.common.constants import DJI_ROCO_DSET_DIR
from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.common.datasets.roco.roco_datasets import ROCODatasets


class DJIROCODatasets(ROCODatasets):
    directory = DJI_ROCO_DSET_DIR

    CentralChina = "robomaster_Central China Regional Competition"
    NorthChina = "robomaster_North China Regional Competition"
    SouthChina = "robomaster_South China Regional Competition"
    Final = "robomaster_Final Tournament"

    @classmethod
    def make_dataset(cls, dataset_name: str, competition_name: str) -> DirectoryROCODataset:
        return DirectoryROCODataset(cls.directory / competition_name, dataset_name)
