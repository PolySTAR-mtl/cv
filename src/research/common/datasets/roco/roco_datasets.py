from abc import abstractmethod
from pathlib import Path
from typing import ClassVar, Iterator, Set

from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.union_dataset import UnionLazyDataset


# FIXME : we should be able to access a builder 2 times
class ROCODatasetsMeta(type):
    def __init__(cls, name: str, bases, dct):
        super().__init__(name, bases, dct)
        cls.__ignore__: Set[str] = set(getattr(cls, "__ignore__", [])) | {"name", "keys", "union"}
        cls.name = cls.__name__.replace("Datasets", "").replace("ROCO", "")

    def __call__(cls, *args, **kwargs):
        raise NotImplemented("This class should not be implemented")

    def __iter__(cls) -> Iterator[ROCODatasetBuilder]:
        return (cls._make_builder_from_name(name) for name in dir(cls) if _is_builder_name(cls, name))

    def __len__(cls):
        return sum(_is_builder_name(cls, name) for name in dir(cls))

    def union(cls) -> UnionLazyDataset[Path, ROCOAnnotation]:
        return UnionLazyDataset(cls, cls.name)

    def __getattribute__(cls, name: str):
        if not _is_builder_name(cls, name):
            return super().__getattribute__(name)

        return cls._make_builder_from_name(name)

    def _make_builder_from_name(cls, name: str) -> ROCODatasetBuilder:
        args = super().__getattribute__(name)
        if not isinstance(args, tuple):
            args = (args,)

        return cls._make_builder_from_args(name, *args)

    @abstractmethod
    def _make_builder_from_args(cls, name: str, *args) -> ROCODatasetBuilder:
        pass


def _is_builder_name(cls: ROCODatasetsMeta, name: str) -> bool:
    return not name.startswith("_") and name not in cls.__ignore__


class ROCODatasets(metaclass=ROCODatasetsMeta):
    main_dir: ClassVar[Path]

    __ignore__ = {"main_dir"}
