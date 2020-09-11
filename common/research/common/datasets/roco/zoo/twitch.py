from typing import Any

from research.common.constants import TWITCH_DSET_DIR
from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.common.datasets.roco.roco_datasets import ROCODatasets


class TwitchROCODatasets(ROCODatasets):
    directory = TWITCH_DSET_DIR / "v1"

    TWITCH_470149568 = ()
    TWITCH_470150052 = ()
    TWITCH_470151286 = ()
    TWITCH_470152289 = ()
    TWITCH_470152730 = ()
    TWITCH_470152838 = ()
    TWITCH_470153081 = ()
    TWITCH_470158483 = ()

    @classmethod
    def make_dataset(cls, dataset_name: str, *args: Any) -> DirectoryROCODataset:
        twitch_id = dataset_name[len("TWITCH_") :]
        return DirectoryROCODataset(cls.directory / twitch_id, f"T{twitch_id}")
