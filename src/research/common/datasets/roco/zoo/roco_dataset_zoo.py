from typing import Iterable, Iterator, Type

from research.common.datasets.roco.roco_datasets import ROCODatasets
from research.common.datasets.roco.zoo.dji import DJIROCODatasets
from research.common.datasets.roco.zoo.dji_zoomed import DJIROCOZoomedDatasets
from research.common.datasets.roco.zoo.twitch import TwitchROCODatasets


class ROCODatasetsZoo(Iterable[Type[ROCODatasets]]):
    DJI_ZOOMED = DJIROCOZoomedDatasets
    DJI = DJIROCODatasets
    TWITCH = TwitchROCODatasets

    DEFAULT_TEST_DATASETS = [TWITCH.T470152838, TWITCH.T470151286]
    DEFAULT_VALIDATION_DATASETS = [TWITCH.T470152932, TWITCH.T470149568]
    TWITCH_TRAIN_DATASETS = [
        TWITCH.T470149066,
        TWITCH.T470150052,
        TWITCH.T470152289,
        TWITCH.T470153081,
        TWITCH.T470158483,
    ]
    DJI_TRAIN_DATASETS = [DJI.FINAL, DJI.CENTRAL_CHINA, DJI.NORTH_CHINA, DJI.SOUTH_CHINA]
    DJI_ZOOMED_TRAIN_DATASETS = [
        DJI_ZOOMED.FINAL,
        DJI_ZOOMED.CENTRAL_CHINA,
        DJI_ZOOMED.NORTH_CHINA,
        DJI_ZOOMED.SOUTH_CHINA,
    ]
    DEFAULT_TRAIN_DATASETS = TWITCH_TRAIN_DATASETS + DJI_TRAIN_DATASETS

    def __iter__(self) -> Iterator[Type[ROCODatasets]]:
        return iter((self.DJI, self.DJI_ZOOMED, self.TWITCH))


ROCODatasetsZoo = ROCODatasetsZoo()
