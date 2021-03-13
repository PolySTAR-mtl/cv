from enum import Enum

from research.constants import TENSORFLOW_RECORDS_DIR


class Records(Enum):
    TWITCH = "Twitch2_Train_T470149066_T470150052_T470152289_T470153081_T470158483_1775_imgs"
    DJI_TWITCH = "Twitch2_Dji_Train_T470149066_T470150052_T470152289_T470153081_T470158483_FINAL_CENTRAL_CHINA_NORTH_CHINA_SOUTH_CHINA_12143_imgs"

    def __init__(self, train_file: str):
        self.train = (TENSORFLOW_RECORDS_DIR / train_file).with_suffix(".record")
        self.val = TENSORFLOW_RECORDS_DIR / "Twitch2_Val_T470152932_T470149568_477_imgs.record"
