from abc import abstractmethod
from enum import Enum, EnumMeta
from pathlib import Path
from typing import Iterator

from polystar.common.utils.str_utils import snake2camel
from research.common.datasets_v3.roco.roco_annotation import ROCOAnnotation
from research.common.datasets_v3.roco.roco_dataset import (
    LazyROCODataset,
    LazyROCOFileDataset,
    ROCODataset,
    ROCOFileDataset,
)
from research.common.datasets_v3.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets_v3.union_dataset import UnionLazyDataset


class ROCODatasets(Enum):
    def __init__(self, dataset_dir_name: str, dataset_name: str = None):
        self.dataset_name = dataset_name or snake2camel(self.name)
        self._dataset_dir_name = dataset_dir_name

    def lazy(self) -> LazyROCODataset:
        return self._dataset_builder.to_images().build_lazy()

    def lazy_files(self) -> LazyROCOFileDataset:
        return self._dataset_builder.build_lazy()

    def dataset(self) -> ROCODataset:
        return self._dataset_builder.to_images().build()

    def files_dataset(self) -> ROCOFileDataset:
        return self._dataset_builder.build()

    @property
    def main_dir(self):
        return self.datasets_dir() / self._dataset_dir_name

    @property
    def _dataset_builder(self) -> ROCODatasetBuilder:
        return ROCODatasetBuilder(self.main_dir, self.dataset_name)

    @classmethod
    @abstractmethod
    def datasets_dir(cls) -> Path:  # Fixme: in python 37, we can define a class var using the _ignore_ attribute
        pass

    @classmethod
    def __iter__(cls) -> Iterator["ROCODatasets"]:  # needed for pycharm typing, dont know why
        return EnumMeta.__iter__(cls)

    @classmethod
    def union(cls) -> UnionLazyDataset[Path, ROCOAnnotation]:
        return UnionLazyDataset((d.lazy_files() for d in cls), cls.datasets_name)

    def __init_subclass__(cls, **kwargs):
        cls.datasets_name = cls.__name__.replace("Datasets", "").replace("ROCO", "")
