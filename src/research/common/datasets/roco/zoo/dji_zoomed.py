from polystar.utils.str_utils import snake2camel
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.roco_datasets import ROCODatasets
from research.constants import DJI_ROCO_ZOOMED_DSET_DIR


class DJIROCOZoomedDatasets(ROCODatasets):
    main_dir = DJI_ROCO_ZOOMED_DSET_DIR

    CENTRAL_CHINA: ROCODatasetBuilder = ()
    NORTH_CHINA: ROCODatasetBuilder = ()
    SOUTH_CHINA: ROCODatasetBuilder = ()
    FINAL: ROCODatasetBuilder = ()

    @classmethod
    def _make_builder_from_args(cls, name: str) -> ROCODatasetBuilder:
        return ROCODatasetBuilder(cls.main_dir / name.lower(), f"{snake2camel(name)}Zoomed")
