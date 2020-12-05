from typing import List, Union

from sklearn.pipeline import Pipeline

from polystar.common.pipeline.classification.classifier_abc import ClassifierABC
from polystar.common.pipeline.pipe_abc import PipeABC, get_pipes_names_without_repetitions
from polystar.common.utils.named_mixin import NamedMixin


class Pipeline(Pipeline, NamedMixin):
    @classmethod
    def from_pipes(cls, pipes: List[Union[PipeABC, ClassifierABC]], name: str = None) -> "Pipeline":
        names = get_pipes_names_without_repetitions(pipes)
        rv = cls(list(zip(names, pipes)))
        rv.name = name or "-".join(names)
        return rv
