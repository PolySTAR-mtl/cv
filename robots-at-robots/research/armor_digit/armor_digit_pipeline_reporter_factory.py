from typing import Iterable

from research.dataset.armor_digit_dataset_factory import ArmorDigitDatasetGenerator
from research_common.dataset.roco_dataset import ROCODataset
from research_common.image_pipeline_evaluation.image_pipeline_evaluation_reporter import ImagePipelineEvaluationReporter
from research_common.image_pipeline_evaluation.image_pipeline_evaluator import ImagePipelineEvaluator


class ArmorDigitPipelineReporterFactory:
    @staticmethod
    def from_roco_datasets(
        train_roco_dataset: ROCODataset,
        test_roco_dataset: ROCODataset,
        acceptable_digits: Iterable[int] = (1, 2, 3, 4, 5, 7),
    ):
        return ImagePipelineEvaluationReporter(
            evaluator=ImagePipelineEvaluator(
                train_roco_dataset=train_roco_dataset,
                test_roco_dataset=test_roco_dataset,
                image_dataset_generator=ArmorDigitDatasetGenerator(set(acceptable_digits)),
            ),
            evaluation_project="armor-digit",
        )
