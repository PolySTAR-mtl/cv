from abc import ABC
from typing import Generic, List, TypeVar

from sklearn.base import BaseEstimator, TransformerMixin

from polystar.common.utils.named_mixin import NamedMixin, get_name

IT = TypeVar("IT")
OT = TypeVar("OT")


class PipeABC(TransformerMixin, BaseEstimator, NamedMixin, Generic[IT, OT], ABC):
    def fit(self, examples: List[IT], labels: List[OT]) -> "PipeABC":
        return self

    def transform(self, examples: List[IT]) -> List[OT]:
        return [self.transform_single(e) for e in examples]

    def transform_single(self, example: IT) -> OT:
        raise NotImplemented("You need to implement either transform or transform_single")


def get_pipes_names_without_repetitions(pipes: List[PipeABC]) -> List[str]:
    names = []
    for pipe in pipes:
        name = get_name(pipe)
        n = names.count(name)
        if n:
            name += str(n + 1)
        names.append(name)
    return names
