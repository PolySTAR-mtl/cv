from pathlib import Path
from typing import Callable, Optional

from optuna import Trial, create_study

from polystar.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.utils.serialization import pkl_dump
from research.robots.evaluation.benchmarker import Benchmarker
from research.robots.evaluation.metrics.accuracy import AccuracyMetric
from research.robots.evaluation.metrics.metric_abc import MetricABC

PipelineFactory = Callable[[Path, Trial], ClassificationPipeline]


class HyperTuner:
    def __init__(self, benchmarker: Benchmarker, metric: MetricABC = AccuracyMetric(), report_frequency: int = 5):
        self.report_frequency = report_frequency
        self.metric = metric
        self.benchmarker = benchmarker
        self._pipeline_factory: Optional[PipelineFactory] = None

    def tune(self, pipeline_factory: PipelineFactory, n_trials: int, minimize: bool = False):
        self._pipeline_factory = pipeline_factory
        study = create_study(direction="minimize" if minimize else "maximize")
        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        self.benchmarker.make_report()
        pkl_dump(study, self.benchmarker.reporter.report_dir / "study")

    def _objective(self, trial: Trial) -> float:
        pipeline = self._pipeline_factory(self.benchmarker.reporter.report_dir, trial)
        performances = self.benchmarker.train_and_evaluate(pipeline)
        if not trial.number % self.report_frequency:
            self.benchmarker.make_report()
        return self.metric(performances.validation.collapse())
