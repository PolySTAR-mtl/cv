from typing import List

from research.common.dataset.directory_roco_dataset import DirectoryROCODataset
from research.robots_at_robots.armor_color.armor_color_dataset import \
    ArmorColorDatasetCache
from research.robots_at_robots.evaluation.image_pipeline_evaluation_reporter import \
    ImagePipelineEvaluationReporter
from research.robots_at_robots.evaluation.image_pipeline_evaluator import \
    ImagePipelineEvaluator


class ArmorColorPipelineReporterFactory:
    @staticmethod
    def from_roco_datasets(
        train_roco_datasets: List[DirectoryROCODataset], test_roco_datasets: List[DirectoryROCODataset]
    ):
        return ImagePipelineEvaluationReporter(
            evaluator=ImagePipelineEvaluator(
                train_roco_datasets=train_roco_datasets,
                test_roco_datasets=test_roco_datasets,
                image_dataset_cache=ArmorColorDatasetCache(),
            ),
            evaluation_project="armor-color",
        )
