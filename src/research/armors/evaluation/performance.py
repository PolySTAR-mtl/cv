from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
from memoized_property import memoized_property

from polystar.filters.filter_abc import FilterABC
from polystar.models.image import FileImage
from polystar.utils.iterable_utils import flatten, group_by
from research.armors.evaluation.set import Set


@dataclass
class ClassificationPerformance:
    examples: List[FileImage]
    labels: np.ndarray
    predictions: np.ndarray
    proba: np.ndarray
    mean_inference_time: float

    @property
    def mistakes(self) -> Sequence[int]:
        return np.where(self.labels != self.predictions)[0]

    @memoized_property
    def unique_labels(self):
        return sorted(set(self.labels) | set(self.predictions))

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class ContextualizedClassificationPerformance(ClassificationPerformance):
    set_: Set
    dataset_name: str
    pipeline_name: str


@dataclass
class ClassificationPerformances(Iterable[ContextualizedClassificationPerformance]):
    performances: List[ContextualizedClassificationPerformance] = field(default_factory=list)

    @property
    def train(self) -> "ClassificationPerformances":
        return self.on_set(Set.TRAIN)

    @property
    def test(self) -> "ClassificationPerformances":
        return self.on_set(Set.TEST)

    @property
    def validation(self) -> "ClassificationPerformances":
        return self.on_set(Set.VALIDATION)

    def on_set(self, set_: Set) -> "ClassificationPerformances":
        return ClassificationPerformances(SetClassificationPerformanceFilter(set_).filter(self.performances))

    def group_by_pipeline(self) -> Dict[str, "ClassificationPerformances"]:
        return {
            name: ClassificationPerformances(performances)
            for name, performances in group_by(self, lambda p: p.pipeline_name).items()
        }

    def collapse(self) -> ClassificationPerformance:
        return ClassificationPerformance(
            examples=flatten(p.examples for p in self),
            labels=np.concatenate([p.labels for p in self]),
            predictions=np.concatenate([p.predictions for p in self]),
            proba=np.concatenate([p.proba for p in self]),
            mean_inference_time=np.average([p.mean_inference_time for p in self], weights=[len(p) for p in self]),
        )

    def __iter__(self):
        return iter(self.performances)

    def __len__(self):
        return len(self.performances)

    def __iadd__(self, other: "ClassificationPerformances"):
        self.performances.extend(other.performances)
        return self


@dataclass
class SetClassificationPerformanceFilter(FilterABC[ContextualizedClassificationPerformance]):
    set_: Set

    def validate_single(self, perf: ContextualizedClassificationPerformance) -> bool:
        return perf.set_ is self.set_
