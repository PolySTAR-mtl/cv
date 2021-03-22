from typing import Iterable, Iterator, List, Type

from research.common.datasets.roco.roco_datasets import ROCODatasets
from research.common.datasets.roco.zoo.dji import DJIROCODatasets
from research.common.datasets.roco.zoo.dji_zoomed import DJIROCOZoomedDatasets
from research.common.datasets.roco.zoo.twitch import TwitchROCODatasets


# FIXME: find a better way to do that (builder need to be instantiated once per call)
# FIXME: improve the singleton pattern here
class ROCODatasetsZooClass(Iterable[Type[ROCODatasets]]):
    DJI_ZOOMED = DJIROCOZoomedDatasets
    DJI = DJIROCODatasets
    TWITCH = TwitchROCODatasets

    @property
    def TWITCH_TRAIN_DATASETS(self) -> List[ROCODatasets]:
        return [
            self.TWITCH.T470149066,
            self.TWITCH.T470150052,
            self.TWITCH.T470152289,
            self.TWITCH.T470153081,
            self.TWITCH.T470158483,
        ]

    @property
    def TWITCH_VALIDATION_DATASETS(self) -> List[ROCODatasets]:
        return [self.TWITCH.T470152932, self.TWITCH.T470149568]

    @property
    def TWITCH_TEST_DATASETS(self) -> List[ROCODatasets]:
        return [self.TWITCH.T470152838, self.TWITCH.T470151286]

    @property
    def DJI_TRAIN_DATASETS(self) -> List[ROCODatasets]:
        return [self.DJI.FINAL, self.DJI.CENTRAL_CHINA, self.DJI.NORTH_CHINA, self.DJI.SOUTH_CHINA]

    @property
    def DJI_ZOOMED_TRAIN_DATASETS(self) -> List[ROCODatasets]:
        return [
            self.DJI_ZOOMED.FINAL,
            self.DJI_ZOOMED.CENTRAL_CHINA,
            self.DJI_ZOOMED.NORTH_CHINA,
            self.DJI_ZOOMED.SOUTH_CHINA,
        ]

    @property
    def TWITCH_DJI_TRAIN_DATASETS(self) -> List[ROCODatasets]:
        return self.TWITCH_TRAIN_DATASETS + self.DJI_TRAIN_DATASETS

    @property
    def TWITCH_DJI_ZOOMED_TRAIN_DATASETS(self) -> List[ROCODatasets]:
        return self.TWITCH_TRAIN_DATASETS + self.DJI_TRAIN_DATASETS

    DEFAULT_TEST_DATASETS = TWITCH_TEST_DATASETS
    DEFAULT_VALIDATION_DATASETS = TWITCH_VALIDATION_DATASETS
    DEFAULT_TRAIN_DATASETS = TWITCH_DJI_TRAIN_DATASETS

    def __iter__(self) -> Iterator[Type[ROCODatasets]]:
        return iter((self.DJI, self.DJI_ZOOMED, self.TWITCH))


ROCODatasetsZoo = ROCODatasetsZooClass()


if __name__ == "__main__":
    ROCODatasetsZoo.DEFAULT_TEST_DATASETS[0].build_lazy()
    ROCODatasetsZoo.DEFAULT_TEST_DATASETS[0].build_lazy()
