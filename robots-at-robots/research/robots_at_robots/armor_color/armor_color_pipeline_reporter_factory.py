from typing import List

from research.common.dataset.directory_roco_dataset import DirectoryROCODataset
from research.common.image_pipeline_evaluation.image_pipeline_evaluation_reporter import \
    ImagePipelineEvaluationReporter
from research.common.image_pipeline_evaluation.image_pipeline_evaluator import \
    ImagePipelineEvaluator
from research.robots_at_robots.armor_color.armor_color_dataset import \
    ArmorColorDatasetGenerator


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
