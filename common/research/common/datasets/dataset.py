from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Generic, Iterable, Iterator, Tuple, TypeVar

from more_itertools import ilen
from polystar.common.utils.misc import identity

ExampleT = TypeVar("ExampleT")
TargetT = TypeVar("TargetT")
ExampleU = TypeVar("ExampleU")
TargetU = TypeVar("TargetU")


class Dataset(Generic[ExampleT, TargetT], Iterable[Tuple[ExampleT, TargetT]], ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def examples(self) -> Iterable[ExampleT]:
        pass

    @property
    @abstractmethod
    def targets(self) -> Iterable[TargetT]:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT]]:
        pass

    @abstractmethod
    def __len__(self):
        pass

    def transform_examples(self, example_transformer: Callable[[ExampleT], ExampleU]) -> "Dataset[ExampleU, TargetT]":
        return self.transform(example_transformer, identity)

    def transform_targets(
        self, target_transformer: Callable[[TargetT], TargetU] = identity
    ) -> "Dataset[ExampleT, TargetU]":
        return self.transform(identity, target_transformer)

    def transform(
        self, example_transformer: Callable[[ExampleT], ExampleU], target_transformer: Callable[[TargetT], TargetU]
    ) -> "Dataset[ExampleU, TargetU]":
        return GeneratorDataset(
            self.name, lambda: ((example_transformer(example), target_transformer(target)) for example, target in self)
        )

    def __str__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    __repr__ = __str__

    def check_consistency(self):
        targets, examples = self.targets, self.examples
        if isinstance(targets, list) and isinstance(examples, list):
            assert len(targets) == len(examples)
        assert ilen(targets) == ilen(examples)


class LazyUnzipper:
    def __init__(self, iterator: Iterator[Tuple]):
        self._iterator = iterator
        self._memory = [deque(), deque()]

    def empty(self, i: int):
        return self._iterator is None and not self._memory[i]

    def elements(self, i: int):
        while True:
            if self._memory[i]:
                yield self._memory[i].popleft()
            elif self._iterator is None:
                return
            else:
                try:
                    elements = next(self._iterator)
                    self._memory[1 - i].append(elements[1 - i])
                    yield elements[i]
                except StopIteration:
                    self._iterator = None
                    return


class LazyDataset(Dataset[ExampleT, TargetT], ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self._unzipper = LazyUnzipper(iter(self))

    @property
    def examples(self) -> Iterable[ExampleT]:
        if self._unzipper.empty(0):
            self._unzipper = LazyUnzipper(iter(self))
        return self._unzipper.elements(0)

    @property
    def targets(self) -> Iterable[ExampleT]:
        if self._unzipper.empty(1):
            self._unzipper = LazyUnzipper(iter(self))
        return self._unzipper.elements(1)

    def __len__(self):
        return ilen(self)


class GeneratorDataset(LazyDataset[ExampleT, TargetT]):
    def __init__(self, name: str, generator: Callable[[], Iterator[Tuple[ExampleT, TargetT]]]):
        self.generator = generator
        super().__init__(name)

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT]]:
        return self.generator()
