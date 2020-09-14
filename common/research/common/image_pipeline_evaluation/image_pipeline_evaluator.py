import logging
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from memoized_property import memoized_property
from polystar.common.image_pipeline.image_pipeline import ImagePipeline
from polystar.common.models.image import Image, load_images
from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.robots_at_robots.dataset.armor_value_dataset import \
    ArmorValueDatasetGenerator
from sklearn.metrics import classification_report, confusion_matrix


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
        image_dataset_generator: ArmorValueDatasetGenerator,
    ):
        logging.info("Loading data")
        self.train_roco_datasets = train_roco_datasets
        self.test_roco_datasets = test_roco_datasets
        (self.train_images_paths, self.train_images, self.train_labels, self.train_dataset_sizes) = load_datasets(
            train_roco_datasets, image_dataset_generator
        )
        (self.test_images_paths, self.test_images, self.test_labels, self.test_dataset_sizes) = load_datasets(
            test_roco_datasets, image_dataset_generator
        )

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


def load_datasets(
    roco_datasets: List[DirectoryROCODataset], image_dataset_generator: ArmorValueDatasetGenerator,
) -> Tuple[List[Path], List[Image], List[Any], List[int]]:
    dataset = image_dataset_generator.from_roco_datasets(roco_datasets)
    dataset_sizes = [len(d) for d in dataset.datasets]

    paths, targets = list(dataset.examples), list(dataset.targets)
    images = list(load_images(paths))
    return paths, images, targets, dataset_sizes
