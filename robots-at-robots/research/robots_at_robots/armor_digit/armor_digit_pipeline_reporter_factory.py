from typing import Iterable, List

from research.common.dataset.directory_roco_dataset import DirectoryROCODataset
from research.common.image_pipeline_evaluation.image_pipeline_evaluation_reporter import ImagePipelineEvaluationReporter
from research.common.image_pipeline_evaluation.image_pipeline_evaluator import ImagePipelineEvaluator
from research.robots_at_robots.dataset import ArmorDigitDatasetGenerator


class ArmorDigitPipelineReporterFactory:
    @staticmethod
    def from_roco_datasets(
        train_roco_datasets: List[DirectoryROCODataset],
        test_roco_datasets: List[DirectoryROCODataset],
        acceptable_digits: Iterable[int] = (1, 2, 3, 4, 5, 7),
    ):
        return ImagePipelineEvaluationReporter(
            evaluator=ImagePipelineEvaluator(
                train_roco_datasets=train_roco_datasets,
                test_roco_datasets=test_roco_datasets,
                image_dataset_generator=ArmorDigitDatasetGenerator(set(acceptable_digits)),
            ),
            evaluation_project="armor-digit",
        )
