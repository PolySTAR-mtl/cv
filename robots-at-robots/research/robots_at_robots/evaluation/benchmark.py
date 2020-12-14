from dataclasses import dataclass
from typing import List

from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from research.common.datasets.image_dataset import FileImageDataset
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.robots_at_robots.dataset.armor_value_dataset_generator import ArmorValueDatasetGenerator
from research.robots_at_robots.evaluation.image_pipeline_evaluation_reporter import ImagePipelineEvaluationReporter
from research.robots_at_robots.evaluation.image_pipeline_evaluator import ImageClassificationPipelineEvaluator
from research.robots_at_robots.evaluation.metrics.f1 import F1Metric
from research.robots_at_robots.evaluation.trainer import ImageClassificationPipelineTrainer


@dataclass
class Benchmarker:
    def __init__(
        self,
        train_datasets: List[FileImageDataset],
        test_datasets: List[FileImageDataset],
        evaluation_project: str,
        experiment_name: str,
        classes: List,
    ):
        self.trainer = ImageClassificationPipelineTrainer(train_datasets)
        self.evaluator = ImageClassificationPipelineEvaluator(train_datasets, test_datasets)
        self.reporter = ImagePipelineEvaluationReporter(
            evaluation_project, experiment_name, classes, other_metrics=[F1Metric()]
        )

    def benchmark(self, pipelines: List[ClassificationPipeline]):
        self.trainer.train_pipelines(pipelines)
        self.reporter.report(self.evaluator.evaluate_pipelines(pipelines))


def make_armor_value_benchmarker(
    train_roco_datasets: List[ROCODatasetBuilder],
    test_roco_datasets: List[ROCODatasetBuilder],
    evaluation_project: str,
    experiment_name: str,
    dataset_generator: ArmorValueDatasetGenerator,
    classes: List,
):
    return Benchmarker(
        dataset_generator.from_roco_datasets(train_roco_datasets),
        dataset_generator.from_roco_datasets(test_roco_datasets),
        evaluation_project=evaluation_project,
        experiment_name=experiment_name,
        classes=classes,
    )
