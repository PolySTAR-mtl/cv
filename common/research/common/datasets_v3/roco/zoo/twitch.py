from research.common.constants import TWITCH_DSET_DIR
from research.common.datasets_v3.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets_v3.roco.roco_datasets import ROCODatasets


class TwitchROCODatasets(ROCODatasets):
    main_dir = TWITCH_DSET_DIR / "v1"

    T470149568: ROCODatasetBuilder = ()
    T470150052: ROCODatasetBuilder = ()
    T470151286: ROCODatasetBuilder = ()
    T470152289: ROCODatasetBuilder = ()
    T470152730: ROCODatasetBuilder = ()
    T470152838: ROCODatasetBuilder = ()
    T470153081: ROCODatasetBuilder = ()
    T470158483: ROCODatasetBuilder = ()

    @classmethod
    def _make_builder_from_args(cls, name: str) -> ROCODatasetBuilder:
        twitch_id = name[1:]
        return ROCODatasetBuilder(cls.main_dir / twitch_id, name)
