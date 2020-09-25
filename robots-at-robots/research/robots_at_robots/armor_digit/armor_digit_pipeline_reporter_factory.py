from typing import Iterable, List

from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.robots_at_robots.armor_digit.armor_digit_dataset import make_armor_digit_dataset_generator
from research.robots_at_robots.evaluation.image_pipeline_evaluation_reporter import ImagePipelineEvaluationReporter
from research.robots_at_robots.evaluation.image_pipeline_evaluator import ImagePipelineEvaluator


class ArmorDigitPipelineReporterFactory:
    @staticmethod
    def from_roco_datasets(
        train_roco_datasets: List[ROCODatasetBuilder],
        test_roco_datasets: List[ROCODatasetBuilder],
        acceptable_digits: Iterable[int] = (1, 2, 3, 4, 5, 7),
    ):
        return ImagePipelineEvaluationReporter(
            evaluator=ImagePipelineEvaluator(
                train_roco_datasets=train_roco_datasets,
                test_roco_datasets=test_roco_datasets,
                image_dataset_generator=make_armor_digit_dataset_generator(acceptable_digits),
            ),
            evaluation_project="armor-digit",
        )
