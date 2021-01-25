from enum import Enum, auto

from sklearn.metrics import f1_score

from research.robots.evaluation.metrics.metric_abc import MetricABC
from research.robots.evaluation.performance import ClassificationPerformance


class F1Strategy(Enum):
    MICRO = auto()
    MACRO = auto()
    SAMPLES = auto()
    WEIGHTED = auto()

    def __repr__(self):
        return self.name.lower()

    __str__ = __repr__


class F1Metric(MetricABC):
    def __init__(self, strategy: F1Strategy = F1Strategy.MACRO):
        self.strategy = strategy

    def __call__(self, performance: ClassificationPerformance) -> float:
        return f1_score(performance.labels, performance.predictions, average=str(self.strategy))

    @property
    def name(self) -> str:
        return f"f1 {self.strategy}"
