from typing import Iterable, Iterator, Type

from research.common.datasets_v3.roco.roco_datasets import ROCODatasets
from research.common.datasets_v3.roco.zoo.dji import DJIROCODatasets
from research.common.datasets_v3.roco.zoo.dji_zoomed import DJIROCOZoomedDatasets
from research.common.datasets_v3.roco.zoo.twitch import TwitchROCODatasets


class ROCODatasetsZoo(Iterable[Type[ROCODatasets]]):
    DJI_ZOOMED = DJIROCOZoomedDatasets
    DJI = DJIROCODatasets
    TWITCH = TwitchROCODatasets

    def __iter__(self) -> Iterator[Type[ROCODatasets]]:
        return iter((self.DJI, self.DJI_ZOOMED, self.TWITCH))


ROCODatasetsZoo = ROCODatasetsZoo()
