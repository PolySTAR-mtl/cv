from abc import ABC
from enum import IntEnum
from typing import ClassVar, Generic, List, Sequence, Tuple, TypeVar

from numpy import asarray, ndarray

from polystar.common.pipeline.classification.classifier_abc import ClassifierABC
from polystar.common.pipeline.pipe_abc import IT, PipeABC
from polystar.common.pipeline.pipeline import Pipeline

EnumT = TypeVar("EnumT", bound=IntEnum)


class ClassificationPipeline(Pipeline, Generic[IT, EnumT], ABC):
    enum: ClassVar[EnumT]

    def __init__(self, steps: List[Tuple[str, PipeABC]]):
        super().__init__(steps)
        self.classifier.n_classes = len(self.enum)

    @property
    def classifier(self) -> ClassifierABC:
        return self.steps[-1][-1]

    def fit(self, x: Sequence[IT], y: List[EnumT], **fit_params):
        y_indices = _labels_to_indices(y)
        return super().fit(x, y_indices, **fit_params)

    def predict(self, x: Sequence[IT]) -> List[EnumT]:
        indices = asarray(self.predict_proba(x)).argmax(axis=1)
        return [self.classes_[i] for i in indices]

    def score(self, x: Sequence[IT], y: List[EnumT], **score_params) -> float:
        """It is needed to have a proper CV"""
        return super().score(x, _labels_to_indices(y), **score_params)

    @property
    def classes_(self) -> List[EnumT]:
        return list(self.enum)

    def __init_subclass__(cls, **kwargs):
        assert hasattr(cls, "enum"), f"You need to provide an `enum` ClassVar for {cls.__name__}"


def _labels_to_indices(labels: List[EnumT]) -> ndarray:
    return asarray([label.value - 1 for label in labels])
