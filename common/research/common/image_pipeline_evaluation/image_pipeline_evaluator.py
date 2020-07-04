import logging
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from memoized_property import memoized_property
from polystar.common.image_pipeline.image_pipeline import ImagePipeline
from polystar.common.models.image import Image
from research.common.dataset.directory_roco_dataset import DirectoryROCODataset
from research.common.image_pipeline_evaluation.image_dataset_generator import ImageDatasetGenerator


@dataclass
class SetClassificationResults:
    labels: np.ndarray
    predictions: np.ndarray
    mean_inference_time: float

    @property
    def report(self) -> Dict:
        return classification_report(self.labels, self.predictions, output_dict=True)

    @property
    def confusion_matrix(self) -> Dict:
        return confusion_matrix(self.labels, self.predictions)

    @property
    def mistakes(self) -> Sequence[int]:
        return np.where(self.labels != self.predictions)[0]

    @memoized_property
    def unique_labels(self) -> List[Any]:
        return sorted(set(self.labels) | set(self.predictions))


@dataclass
class ClassificationResults:
    train_results: SetClassificationResults
    test_results: SetClassificationResults
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
        (
            self.train_images_paths,
            self.train_images,
            self.train_labels,
            self.train_dataset_sizes,
        ) = image_dataset_generator.from_roco_datasets(train_roco_datasets)
        (
            self.test_images_paths,
            self.test_images,
            self.test_labels,
            self.test_dataset_sizes,
        ) = image_dataset_generator.from_roco_datasets(test_roco_datasets)

    def evaluate_pipelines(self, pipelines: Iterable[ImagePipeline]) -> Dict[str, ClassificationResults]:
        return {str(pipeline): self.evaluate(pipeline) for pipeline in pipelines}

    def evaluate(self, pipeline: ImagePipeline) -> ClassificationResults:
        logging.info(f"Training pipeline {pipeline}")
        pipeline.fit(self.train_images, self.train_labels)

        logging.info(f"Infering")
        train_results = self._evaluate_on_set(pipeline, self.train_images, self.train_labels)
        test_results = self._evaluate_on_set(pipeline, self.test_images, self.test_labels)

        return ClassificationResults(
            train_results=train_results, test_results=test_results, full_pipeline_name=repr(pipeline),
        )

    @staticmethod
    def _evaluate_on_set(pipeline: ImagePipeline, images: List[Image], labels: List[Any]) -> SetClassificationResults:
        t = time()
        preds = pipeline.predict(images)
        mean_time = (time() - t) / len(images)
        return SetClassificationResults(np.asarray(labels), np.asarray(preds), mean_time)
