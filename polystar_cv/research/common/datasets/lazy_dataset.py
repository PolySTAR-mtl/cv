from abc import ABC, abstractmethod
from typing import Generic, Iterable, Iterator, Tuple, TypeVar

ExampleT = TypeVar("ExampleT")
TargetT = TypeVar("TargetT")


class LazyDataset(Generic[ExampleT, TargetT], Iterable[Tuple[ExampleT, TargetT, str]], ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT, str]]:
        pass

    def __len__(self):
        raise NotImplemented()

    def __str__(self):
        return f"dataset {self.name}"

    __repr__ = __str__
