from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from polystar.common.models.image import Image
from polystar.common.models.trt_model import TRTModel
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_objects_factory import (DetectedObjectFactory,
                                                                                       ObjectParams)
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.common.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC


@dataclass
class TRTModelObjectsDetector(ObjectsDetectorABC):
    trt_model: TRTModel

    def detect(self, image: Image) -> Tuple[List[DetectedRobot], List[DetectedArmor]]:
        results = self.trt_model(image)
        return self._construct_objects_from_trt_results(results, image)

    def _construct_objects_from_trt_results(
        self, results: np.ndarray, image: Image
    ) -> Tuple[List[DetectedRobot], List[DetectedArmor]]:
        return self.objects_factory.make_lists(
            [
                ObjectParams(
                    ymin=float(ymin),
                    xmin=float(xmin),
                    ymax=float(ymax),
                    xmax=float(xmax),
                    score=float(score),
                    object_class_id=int(object_class_id),
                )
                for (_, object_class_id, score, xmin, ymin, xmax, ymax) in results
                if object_class_id >= 0
            ],
            image,
        )
