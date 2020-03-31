import logging
from dataclasses import dataclass
from time import time
from typing import List, Tuple, Dict, Any, Iterable

from sklearn.metrics import classification_report, confusion_matrix

from polystar.common.image_pipeline.image_pipeline import ImagePipeline
from polystar.common.models.image import Image
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset
from research_common.image_pipeline_evaluation.image_dataset_generator import ImageDatasetGenerator


@dataclass
class ClassificationResults:
    train_report: Dict
    train_confusion_matrix: Dict
    train_mean_inference_time: float
    test_report: Dict
    test_confusion_matrix: Dict
    test_mean_inference_time: float
    full_pipeline_name: str


class ImagePipelineEvaluator:
    def __init__(
        self,
        train_roco_datasets: List[DirectoryROCODataset],
        test_roco_datasets: List[DirectoryROCODataset],
        image_dataset_generator: ImageDatasetGenerator,
    ):
        logging.info("Loading data")
        self.train_roco_datasets = train_roco_datasets
        self.test_roco_datasets = test_roco_datasets
        self.train_images, self.train_labels, self.train_dataset_sizes = image_dataset_generator.from_roco_datasets(
            train_roco_datasets
        )
        self.test_images, self.test_labels, self.test_dataset_sizes = image_dataset_generator.from_roco_datasets(
            test_roco_datasets
        )

    def evaluate_pipelines(self, pipelines: Iterable[ImagePipeline]) -> Dict[str, ClassificationResults]:
        return {str(pipeline): self.evaluate(pipeline) for pipeline in pipelines}

    def evaluate(self, pipeline: ImagePipeline) -> ClassificationResults:
        logging.info(f"Training pipeline {pipeline}")
        pipeline.fit(self.train_images, self.train_labels)

        logging.info(f"Infering")
        train_report, train_confusion_matrix, train_time = self._evaluate_on_set(
            pipeline, self.train_images, self.train_labels
        )
        test_report, test_confusion_matrix, test_time = self._evaluate_on_set(
            pipeline, self.test_images, self.test_labels
        )

        return ClassificationResults(
            train_report=train_report,
            test_report=test_report,
            train_mean_inference_time=train_time,
            test_mean_inference_time=test_time,
            train_confusion_matrix=train_confusion_matrix,
            test_confusion_matrix=test_confusion_matrix,
            full_pipeline_name=repr(pipeline),
        )

    @staticmethod
    def _evaluate_on_set(pipeline: ImagePipeline, images: List[Image], labels: List[Any]) -> Tuple[Dict, Dict, float]:
        t = time()
        preds = pipeline.predict(images)
        mean_time = (time() - t) / len(images)
        return classification_report(labels, preds, output_dict=True), confusion_matrix(labels, preds), mean_time
