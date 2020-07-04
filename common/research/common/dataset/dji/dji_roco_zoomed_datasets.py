from enum import Enum, auto

from polystar.common.utils.str_utils import camel2snake
from research.common.constants import DJI_ROCO_ZOOMED_DSET_DIR
from research.common.dataset.directory_roco_dataset import DirectoryROCODataset


class DJIROCOZoomedDataset(DirectoryROCODataset, Enum):
    def __init__(self, _):
        super().__init__(DJI_ROCO_ZOOMED_DSET_DIR / camel2snake(self.name), f"{self.name}ZoomedV2")

    CentralChina = auto()
    NorthChina = auto()
    SouthChina = auto()
    Final = auto()
