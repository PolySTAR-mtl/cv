from typing import Any, Iterator, List, Tuple

from research.common.dataset.directory_roco_dataset import DirectoryROCODataset


class ROCODatasets:
    def make_dataset(dataset_name: str, *args: Any) -> DirectoryROCODataset:
        pass

    def __init_subclass__(cls, **kwargs):
        cls.datasets: List[DirectoryROCODataset] = []
        for dataset_name, args in cls.__dict__.items():
            if (
                not callable(args)
                and not dataset_name.startswith("_")
                and dataset_name not in ("make_dataset", "datasets")
            ):
                if not isinstance(args, Tuple):
                    args = (args,)
                dataset = cls.make_dataset(dataset_name, *args)
                setattr(cls, dataset_name, dataset)
                cls.datasets.append(dataset)

    def __iter__(self) -> Iterator[DirectoryROCODataset]:
        return self.datasets.__iter__()
