from abc import abstractmethod
from enum import Enum
from pathlib import Path

from research.constants import TENSORFLOW_RECORDS_DIR


class Records(Enum):
    @property
    @abstractmethod
    def val(self) -> Path:
        pass

    @property
    def task_name(self) -> str:
        return self.__class__.__name__.replace("Records", "").lower()

    def __init__(self, train_file: str):
        self.train = (TENSORFLOW_RECORDS_DIR / train_file).with_suffix(".record")


class ROCORecords(Records):
    TWITCH = "Twitch2_Train_T470149066_T470150052_T470152289_T470153081_T470158483_1775_imgs"
    DJI_TWITCH = "Twitch2_Dji_Train_T470149066_T470150052_T470152289_T470153081_T470158483_FINAL_CENTRAL_CHINA_NORTH_CHINA_SOUTH_CHINA_12143_imgs"

    @property
    def val(self) -> Path:
        return TENSORFLOW_RECORDS_DIR / "Twitch2_Val_T470152932_T470149568_477_imgs.record"


class AIRRecords(Records):
    TWITCH = "air/Twitch2_Train_T470149066_AIR_T470150052_AIR_T470152289_AIR_T470153081_AIR_T470158483_AIR_2837_imgs"

    @property
    def val(self) -> Path:
        return TENSORFLOW_RECORDS_DIR / "air/Twitch2_Val_T470152932_AIR_T470149568_AIR_793_imgs"
