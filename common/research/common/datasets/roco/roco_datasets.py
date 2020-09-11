from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, List, Tuple

from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset


class ROCODatasets(Iterable[DirectoryROCODataset]):
    name: ClassVar[str]
    datasets: ClassVar[List[DirectoryROCODataset]]
    directory: ClassVar[Path]

    @classmethod
    @abstractmethod
    def make_dataset(cls, dataset_name: str, *args: Any) -> DirectoryROCODataset:
        pass

    def __init_subclass__(cls, **kwargs):
        cls.datasets: List[DirectoryROCODataset] = []
        for dataset_name, args in cls.__dict__.items():
            if not dataset_name.islower():
                if not isinstance(args, Tuple):
                    args = (args,)
                dataset = cls.make_dataset(dataset_name, *args)
                setattr(cls, dataset_name, dataset)
                cls.datasets.append(dataset)

        cls.name = cls.__name__[: -len("ROCODatasets")]

    def __iter__(self) -> Iterator[DirectoryROCODataset]:
        return self.datasets.__iter__()
