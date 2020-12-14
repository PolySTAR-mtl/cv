from enum import Enum
from itertools import chain
from time import time
from typing import Generic, Iterable, List

import numpy as np

from polystar.common.models.image import file_images_to_images
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.utils.iterable_utils import flatten
from research.common.datasets.image_dataset import FileImageDataset
from research.common.datasets.lazy_dataset import TargetT
from research.robots_at_robots.evaluation.performance import (
    ClassificationPerformance,
    ClassificationPerformances,
    ContextualizedClassificationPerformance,
)
from research.robots_at_robots.evaluation.set import Set


class ImageClassificationPipelineEvaluator(Generic[TargetT]):
    def __init__(
        self, train_datasets: List[FileImageDataset], test_datasets: List[FileImageDataset],
    ):
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def evaluate_pipelines(self, pipelines: Iterable[ClassificationPipeline]) -> ClassificationPerformances:
        return ClassificationPerformances(flatten(self._evaluate_pipeline(pipeline) for pipeline in pipelines))

    def _evaluate_pipeline(self, pipeline: ClassificationPipeline) -> Iterable[ContextualizedClassificationPerformance]:
        return chain(
            self._evaluate_pipeline_on_set(pipeline, self.train_datasets, Set.TRAIN),
            self._evaluate_pipeline_on_set(pipeline, self.test_datasets, Set.TEST),
        )

    @staticmethod
    def _evaluate_pipeline_on_set(
        pipeline: ClassificationPipeline, datasets: List[FileImageDataset], set_: Set
    ) -> Iterable[ContextualizedClassificationPerformance]:
        for dataset in datasets:
            t = time()
            proba, classes = pipeline.predict_proba_and_classes(file_images_to_images(dataset.examples))
            mean_time = (time() - t) / len(dataset)
            yield ContextualizedClassificationPerformance(
                examples=dataset.examples,
                labels=_labels_to_numpy(dataset.targets),
                predictions=_labels_to_numpy(classes),
                proba=proba,
                mean_inference_time=mean_time,
                set_=set_,
                dataset_name=dataset.name,
                pipeline_name=pipeline.name,
            )


def _labels_to_numpy(labels: List[Enum]) -> np.ndarray:
    return np.asarray([str(label) for label in labels])
