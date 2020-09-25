from research.common.constants import DJI_ROCO_DSET_DIR
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.roco_datasets import ROCODatasets


class DJIROCODatasets(ROCODatasets):
    main_dir = DJI_ROCO_DSET_DIR

    CENTRAL_CHINA: ROCODatasetBuilder = "robomaster_Central China Regional Competition"
    NORTH_CHINA: ROCODatasetBuilder = "robomaster_North China Regional Competition"
    SOUTH_CHINA: ROCODatasetBuilder = "robomaster_South China Regional Competition"
    FINAL: ROCODatasetBuilder = "robomaster_Final Tournament"

    @classmethod
    def _make_builder_from_args(cls, name: str, competition_name: str) -> ROCODatasetBuilder:
        return ROCODatasetBuilder(cls.main_dir / competition_name, name)
