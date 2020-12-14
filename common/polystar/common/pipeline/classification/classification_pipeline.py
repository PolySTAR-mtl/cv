from abc import ABC
from enum import IntEnum
from typing import ClassVar, Generic, List, Sequence, Tuple, TypeVar

from numpy import asarray, ndarray, pad

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

    def fit(self, x: Sequence[IT], y: List[EnumT], validation_size: int = 0, **fit_params):
        if isinstance(self.classifier, ClassifierABC):
            fit_params[f"{self.classifier.__class__.__name__}__validation_size"] = validation_size
        y_indices = _labels_to_indices(y)
        return super().fit(x, y_indices, **fit_params)

    def predict(self, x: Sequence[IT]) -> List[EnumT]:
        return self.predict_proba_and_classes(x)[1]

    def predict_proba(self, x: Sequence[IT]) -> ndarray:
        proba = super().predict_proba(x)
        missing_classes = self.classifier.n_classes - proba.shape[1]
        if not missing_classes:
            return proba
        return pad(proba, ((0, 0), (0, missing_classes)))

    def predict_proba_and_classes(self, x: Sequence[IT]) -> Tuple[ndarray, List[EnumT]]:
        proba = asarray(self.predict_proba(x))
        indices = proba.argmax(axis=1)
        classes = [self.classes_[i] for i in indices]
        return proba, classes

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
