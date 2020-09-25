from typing import List

from research.common.datasets_v3.roco.roco_dataset_builder import ROCODatasetBuilder
from research.robots_at_robots.armor_color.armor_color_dataset import make_armor_color_dataset_generator
from research.robots_at_robots.evaluation.image_pipeline_evaluation_reporter import ImagePipelineEvaluationReporter
from research.robots_at_robots.evaluation.image_pipeline_evaluator import ImagePipelineEvaluator


class ArmorColorPipelineReporterFactory:
    @staticmethod
    def from_roco_datasets(train_roco_datasets: List[ROCODatasetBuilder], test_roco_datasets: List[ROCODatasetBuilder]):
        return ImagePipelineEvaluationReporter(
            evaluator=ImagePipelineEvaluator(
                train_roco_datasets=train_roco_datasets,
                test_roco_datasets=test_roco_datasets,
                image_dataset_generator=make_armor_color_dataset_generator(),
            ),
            evaluation_project="armor-color",
        )
