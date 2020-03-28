from enum import Enum

from research_common.constants import TWITCH_DSET_DIR
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset


class TwitchROCODataset(DirectoryROCODataset, Enum):
    def __init__(self, competition_name: str):
        super().__init__(TWITCH_DSET_DIR / "v1" / competition_name, self.name)

    TWITCH_470149568 = "470149568"
    TWITCH_470150052 = "470150052"
    TWITCH_470151286 = "470151286"
    TWITCH_470152289 = "470152289"
    TWITCH_470152730 = "470152730"
    TWITCH_470152838 = "470152838"
    TWITCH_470153081 = "470153081"
    TWITCH_470158483 = "470158483"
