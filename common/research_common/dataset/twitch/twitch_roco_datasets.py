"""
>>> TwitchROCODataset.TWITCH_470149568.dataset_name
'T470149568'

>>> from research_common.constants import DSET_DIR
>>> TwitchROCODataset.TWITCH_470149568.dataset_path.relative_to(DSET_DIR)
PosixPath('twitch/v1/470149568')

>>> TwitchROCODataset.TWITCH_470149568.video_url
'https://www.twitch.tv/videos/470149568'
"""

from enum import Enum, auto

from research_common.constants import TWITCH_DSET_DIR
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset


class TwitchROCODataset(DirectoryROCODataset, Enum):
    def __init__(self, _):
        self.twitch_id = self.name[len("TWITCH_") :]
        super().__init__(TWITCH_DSET_DIR / "v1" / self.twitch_id, f"T{self.twitch_id}")

    @property
    def video_url(self) -> str:
        return f"https://www.twitch.tv/videos/{self.twitch_id}"

    TWITCH_470149568 = auto()
    TWITCH_470150052 = auto()
    TWITCH_470151286 = auto()
    TWITCH_470152289 = auto()
    TWITCH_470152730 = auto()
    TWITCH_470152838 = auto()
    TWITCH_470153081 = auto()
    TWITCH_470158483 = auto()
