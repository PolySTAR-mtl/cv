from enum import auto
from pathlib import Path

from polystar.common.utils.str_utils import snake2camel
from research.common.constants import DJI_ROCO_ZOOMED_DSET_DIR
from research.common.datasets_v3.roco.roco_datasets import ROCODatasets


class DJIROCOZoomedDatasets(ROCODatasets):
    def __init__(self, _):
        super().__init__(self.name.lower(), f"{snake2camel(self.name)}ZoomedV2")

    CENTRAL_CHINA = auto()
    NORTH_CHINA = auto()
    SOUTH_CHINA = auto()
    FINAL = auto()

    @classmethod
    def datasets_dir(cls) -> Path:
        return DJI_ROCO_ZOOMED_DSET_DIR
