from enum import auto
from pathlib import Path

from research.common.constants import TWITCH_DSET_DIR
from research.common.datasets_v3.roco.roco_datasets import ROCODatasets


class TwitchROCODatasets(ROCODatasets):
    def __init__(self, _):
        super().__init__(self.twitch_id)

    T470149568 = auto()
    T470150052 = auto()
    T470151286 = auto()
    T470152289 = auto()
    T470152730 = auto()
    T470152838 = auto()
    T470153081 = auto()
    T470158483 = auto()

    @classmethod
    def datasets_dir(cls) -> Path:
        return TWITCH_DSET_DIR / "v1"

    @property
    def twitch_id(self) -> str:
        return self.name[len("T") :]

    @property
    def video_url(self) -> str:
        return f"https://www.twitch.tv/videos/{self.twitch_id}"
