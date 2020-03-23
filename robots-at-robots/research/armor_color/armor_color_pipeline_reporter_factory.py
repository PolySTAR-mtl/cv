from research.dataset.armor_color_dataset_factory import ArmorColorDatasetGenerator
from research_common.dataset.roco_dataset import ROCODataset
from research_common.image_pipeline_evaluation.image_pipeline_evaluation_reporter import ImagePipelineEvaluationReporter
from research_common.image_pipeline_evaluation.image_pipeline_evaluator import ImagePipelineEvaluator


class ArmorColorPipelineReporterFactory:
    @staticmethod
    def from_roco_datasets(train_roco_dataset: ROCODataset, test_roco_dataset: ROCODataset):
        return ImagePipelineEvaluationReporter(
            evaluator=ImagePipelineEvaluator(
                train_roco_dataset=train_roco_dataset,
                test_roco_dataset=test_roco_dataset,
                image_dataset_generator=ArmorColorDatasetGenerator(),
            ),
            evaluation_project="armor-color",
        )
