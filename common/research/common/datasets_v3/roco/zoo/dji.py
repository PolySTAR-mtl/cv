from pathlib import Path

from research.common.constants import DJI_ROCO_DSET_DIR
from research.common.datasets_v3.roco.roco_datasets import ROCODatasets


class DJIROCODatasets(ROCODatasets):
    CENTRAL_CHINA = "robomaster_Central China Regional Competition"
    NORTH_CHINA = "robomaster_North China Regional Competition"
    SOUTH_CHINA = "robomaster_South China Regional Competition"
    FINAL = "robomaster_Final Tournament"

    @classmethod
    def datasets_dir(cls) -> Path:
        return DJI_ROCO_DSET_DIR
