from typing import Any

from research.common.constants import TWITCH_DSET_DIR
from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.common.datasets.roco.roco_datasets import ROCODatasets


class TwitchROCODatasets(ROCODatasets):
    directory = TWITCH_DSET_DIR / "v1"

    T470149568 = ()
    T470150052 = ()
    T470151286 = ()
    T470152289 = ()
    T470152730 = ()
    T470152838 = ()
    T470153081 = ()
    T470158483 = ()

    @classmethod
    def make_dataset(cls, dataset_name: str, *args: Any) -> DirectoryROCODataset:
        twitch_id = dataset_name[len("T") :]
        return DirectoryROCODataset(cls.directory / twitch_id, dataset_name)
