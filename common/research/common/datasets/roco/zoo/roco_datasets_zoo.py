from typing import Iterable

from research.common.datasets.roco.roco_datasets import ROCODatasets
from research.common.datasets.roco.zoo.dji import DJIROCODatasets
from research.common.datasets.roco.zoo.dji_zoomed import DJIROCOZoomedDatasets
from research.common.datasets.roco.zoo.twitch import TwitchROCODatasets


class ROCODatasetsZoo(Iterable[ROCODatasets]):
    DJI_ZOOMED = DJIROCOZoomedDatasets()
    DJI = DJIROCODatasets()
    TWITCH = TwitchROCODatasets()

    def __iter__(self):
        return (self.DJI, self.DJI_ZOOMED, self.TWITCH).__iter__()
