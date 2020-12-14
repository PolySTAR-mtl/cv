from enum import Enum
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
        self,
        train_datasets: List[FileImageDataset],
        validation_datasets: List[FileImageDataset],
        test_datasets: List[FileImageDataset],
    ):
        self.set2datasets = {Set.TRAIN: train_datasets, Set.VALIDATION: validation_datasets, Set.TEST: test_datasets}

    def evaluate_pipelines(self, pipelines: Iterable[ClassificationPipeline]) -> ClassificationPerformances:
        return ClassificationPerformances(flatten(self._evaluate_pipeline(pipeline) for pipeline in pipelines))

    def _evaluate_pipeline(self, pipeline: ClassificationPipeline) -> Iterable[ContextualizedClassificationPerformance]:
        for set_ in Set:
            yield from self._evaluate_pipeline_on_set(pipeline, set_)

    def _evaluate_pipeline_on_set(
        self, pipeline: ClassificationPipeline, set_: Set
    ) -> Iterable[ContextualizedClassificationPerformance]:
        for dataset in self.set2datasets[set_]:
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
