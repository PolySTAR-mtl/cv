from typing import List, Union

from sklearn.pipeline import Pipeline

from polystar.pipeline.classification.classifier_abc import ClassifierABC
from polystar.pipeline.pipe_abc import PipeABC, get_pipes_names_without_repetitions
from polystar.utils.named_mixin import NamedMixin

Pipes = List[Union[PipeABC, ClassifierABC]]


class Pipeline(Pipeline, NamedMixin):
    @classmethod
    def from_pipes(cls, pipes: Pipes, name: str = None) -> "Pipeline":
        names = get_pipes_names_without_repetitions(pipes)
        rv = cls(list(zip(names, pipes)))
        rv.name = name or "-".join(names)
        return rv
