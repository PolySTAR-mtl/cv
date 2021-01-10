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
    DEFAULT_VALIDATION_DATASETS = [TWITCH.T470152838, TWITCH.T470151286]
    TWITCH_TRAIN_DATASETS = [TWITCH.T470150052, TWITCH.T470152730, TWITCH.T470153081, TWITCH.T470158483]
    DEFAULT_TRAIN_DATASETS = TWITCH_TRAIN_DATASETS + [DJI.FINAL, DJI.CENTRAL_CHINA, DJI.NORTH_CHINA, DJI.SOUTH_CHINA]

    def __iter__(self) -> Iterator[Type[ROCODatasets]]:
        return iter((self.DJI, self.DJI_ZOOMED, self.TWITCH))


ROCODatasetsZoo = ROCODatasetsZoo()
