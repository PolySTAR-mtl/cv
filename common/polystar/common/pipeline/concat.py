from typing import List

from sklearn.pipeline import FeatureUnion

from polystar.common.pipeline.pipe_abc import PipeABC, get_pipes_names_without_repetitions
from polystar.common.utils.named_mixin import NamedMixin


class Concat(FeatureUnion, NamedMixin):
    @staticmethod
    def from_pipes(pipes: List[PipeABC], name: str = "Concat") -> "Concat":
        names = get_pipes_names_without_repetitions(pipes)
        rv = Concat(list(zip(names, pipes)))
        rv.name = name
        return rv
