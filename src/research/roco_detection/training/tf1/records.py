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

    def __init__(self, record_name: str):
        self.train = self.records_path(record_name)

    def records_path(self, record_name):
        return TENSORFLOW_RECORDS_DIR / self.task_name / record_name / "*.record"


class ROCORecords(Records):
    TWITCH = "Twitch2_Train_T470149066_T470150052_T470152289_T470153081_T470158483_T470152730_1891_imgs"
    DJI = "Dji_CENTRAL_CHINA_FINAL_NORTH_CHINA_SOUTH_CHINA_10368_imgs"

    @property
    def val(self) -> Path:
        return self.records_path("Twitch2_Val_T470152932_T470149568_477_imgs")


class AIRRecords(Records):
    TWITCH = "Twitch2_Train_T470149066_AIR_T470150052_AIR_T470152289_AIR_T470153081_AIR_T470158483_AIR_2837_imgs"
    DJI = "Dji_CENTRAL_CHINA_AIR_FINAL_AIR_NORTH_CHINA_AIR_SOUTH_CHINA_AIR_66750_imgs"

    @property
    def val(self) -> Path:
        return self.records_path("Twitch2_Val_T470152932_AIR_T470149568_AIR_793_imgs")
