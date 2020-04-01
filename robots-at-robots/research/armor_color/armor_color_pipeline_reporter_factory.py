from typing import List

from research.dataset.armor_color_dataset_factory import ArmorColorDatasetGenerator
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset
from research_common.image_pipeline_evaluation.image_pipeline_evaluation_reporter import ImagePipelineEvaluationReporter
from research_common.image_pipeline_evaluation.image_pipeline_evaluator import ImagePipelineEvaluator


class ArmorColorPipelineReporterFactory:
    @staticmethod
    def from_roco_datasets(
        train_roco_datasets: List[DirectoryROCODataset], test_roco_datasets: List[DirectoryROCODataset]
    ):
        return ImagePipelineEvaluationReporter(
            evaluator=ImagePipelineEvaluator(
                train_roco_datasets=train_roco_datasets,
                test_roco_datasets=test_roco_datasets,
                image_dataset_generator=ArmorColorDatasetGenerator(),
            ),
            evaluation_project="armor-color",
        )
