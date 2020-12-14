from research.robots_at_robots.evaluation.metrics.metric_abc import MetricABC
from research.robots_at_robots.evaluation.performance import ClassificationPerformance


class AccuracyMetric(MetricABC):
    def __call__(self, performance: ClassificationPerformance) -> float:
        return (performance.labels == performance.predictions).mean()

    @property
    def name(self) -> str:
        return "accuracy"
