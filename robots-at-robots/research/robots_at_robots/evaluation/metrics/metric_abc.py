from abc import ABC, abstractmethod

from research.robots_at_robots.evaluation.performance import ClassificationPerformance


class MetricABC(ABC):
    @abstractmethod
    def __call__(self, performance: ClassificationPerformance) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __repr__(self):
        return self.name
