from typing import Any, Tuple

from research.common.dataset.directory_roco_dataset import DirectoryROCODataset


class ROCODatasets:
    def _make_dataset(dataset_name: str, *args: Any) -> DirectoryROCODataset:
        pass

    def __init_subclass__(cls, **kwargs):
        for dataset_name, args in cls.__dict__.items():
            if not callable(args) and not dataset_name.startswith("_"):
                if not isinstance(args, Tuple):
                    args = (args,)
                setattr(cls, dataset_name, cls._make_dataset(dataset_name, *args))
