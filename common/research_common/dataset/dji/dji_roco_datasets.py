from enum import Enum

from research_common.constants import ROCO_DSET_DIR
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset


class DJIROCODataset(DirectoryROCODataset, Enum):
    def __init__(self, competition_name: str):
        super().__init__(ROCO_DSET_DIR / competition_name, self.name)

    CentralChina = "robomaster_Central China Regional Competition"
    NorthChina = "robomaster_North China Regional Competition"
    SouthChina = "robomaster_South China Regional Competition"
    Final = "robomaster_Final Tournament"
