from typing import List

from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.robots_at_robots.armor_digit.armor_digit_dataset import make_armor_digit_dataset_generator
from research.robots_at_robots.evaluation.image_pipeline_evaluation_reporter import ImagePipelineEvaluationReporter
from research.robots_at_robots.evaluation.image_pipeline_evaluator import ImagePipelineEvaluator


class ArmorDigitPipelineReporterFactory:
    @staticmethod
    def from_roco_datasets(
        train_roco_datasets: List[ROCODatasetBuilder],
        test_roco_datasets: List[ROCODatasetBuilder],
        experiment_name: str,
    ):
        return ImagePipelineEvaluationReporter(
            evaluator=ImagePipelineEvaluator(
                train_roco_datasets=train_roco_datasets,
                test_roco_datasets=test_roco_datasets,
                image_dataset_generator=make_armor_digit_dataset_generator(),
            ),
            evaluation_project="armor-digit",
            experiment_name=experiment_name,
        )
