import logging
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import List, Sequence

from polystar.pipeline.classification.classification_pipeline import ClassificationPipeline
from research.armors.evaluation.evaluator import ImageClassificationPipelineEvaluator
from research.armors.evaluation.metrics.f1 import F1Metric
from research.armors.evaluation.performance import ClassificationPerformances
from research.armors.evaluation.reporter import ImagePipelineEvaluationReporter
from research.armors.evaluation.trainer import ImageClassificationPipelineTrainer
from research.common.datasets.image_dataset import FileImageDataset

logger = logging.getLogger(__name__)


@dataclass
class Benchmarker:
    def __init__(
        self,
        train_datasets: List[FileImageDataset],
        validation_datasets: List[FileImageDataset],
        test_datasets: List[FileImageDataset],
        classes: List,
        report_dir: Path,
    ):
        report_dir.mkdir(exist_ok=True, parents=True)
        self.trainer = ImageClassificationPipelineTrainer(train_datasets, validation_datasets)
        self.evaluator = ImageClassificationPipelineEvaluator(train_datasets, validation_datasets, test_datasets)
        self.reporter = ImagePipelineEvaluationReporter(
            report_dir=report_dir, classes=classes, other_metrics=[F1Metric()]
        )
        self.performances = ClassificationPerformances()
        logger.info(f"Run `tensorboard --logdir={report_dir}` for realtime logs when using keras")

    def train_and_evaluate(self, pipeline: ClassificationPipeline) -> ClassificationPerformances:
        self.trainer.train_pipeline(pipeline)
        pipeline_performances = self.evaluator.evaluate_pipeline(pipeline)
        self.performances += pipeline_performances
        return pipeline_performances

    def benchmark(
        self, pipelines: Sequence[ClassificationPipeline] = (), trained_pipelines: Sequence[ClassificationPipeline] = ()
    ):
        self.trainer.train_pipelines(pipelines)
        self.performances += self.evaluator.evaluate_pipelines(chain(pipelines, trained_pipelines))
        self.make_report()

    def make_report(self):
        self.reporter.report(self.performances)
