from typing import List

import numpy as np

from polystar.common.models.image import Image
from polystar.common.models.label_map import LabelMap
from polystar.common.models.trt_model import TRTModel
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.detected_objects.detected_objects_factory import DetectedObjectFactory
from polystar.common.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC


class TRTModelObjectsDetector(ObjectsDetectorABC):
    def __init__(self, trt_model: TRTModel, label_map: LabelMap):
        self.label_map = label_map
        self.trt_model = trt_model

    def detect(self, image: Image) -> List[DetectedObject]:
        results = self.trt_model(image)
        return self._construct_objects_from_trt_results(results, image)

    def _construct_objects_from_trt_results(self, results: np.ndarray, image: Image) -> List[DetectedObject]:
        objects_factory = DetectedObjectFactory(image, self.label_map)
        return [
            objects_factory.from_relative_positions(
                ymin=float(ymin),
                xmin=float(xmin),
                ymax=float(ymax),
                xmax=float(xmax),
                score=float(score),
                object_class_id=int(object_class_id),
            )
            for (_, object_class_id, score, xmin, ymin, xmax, ymax) in results
            if object_class_id >= 0
        ]
