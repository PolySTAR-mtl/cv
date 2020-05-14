from enum import Enum
from typing import List

import numpy as np

from polystar.common.models.box import Box
from polystar.common.models.image import Image
from polystar.common.models.label_map import LabelMap
from polystar.common.models.object import Object, ObjectType
from polystar.common.models.trt_model import TRTModel
from polystar.common.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC


class TRTModelObjectsDetector(ObjectsDetectorABC):
    def __init__(self, trt_model: TRTModel, label_map: LabelMap):
        self.label_map = label_map
        self.trt_model = trt_model

    def detect(self, image: Image) -> List[Object]:
        results = self.trt_model(image)
        return self._construct_objects_from_trt_results(results, image)

    def _construct_object_from_trt_result(self, result: List[float], image_height: int, image_width: int):
        xmin = TRTResultGetters.X_MIN.get_value(result)
        xmax = TRTResultGetters.X_MAX.get_value(result)
        ymin = TRTResultGetters.Y_MIN.get_value(result)
        ymax = TRTResultGetters.Y_MAX.get_value(result)
        return Object(
            type=ObjectType(self.label_map.name_of(TRTResultGetters.CLS.get_value(result))),
            confidence=TRTResultGetters.CONF.get_value(result),
            box=Box.from_positions(
                x1=int(xmin * image_width),
                y1=int(ymin * image_height),
                x2=int(xmax * image_width),
                y2=int(ymax * image_height),
            ),
        )

    def _construct_objects_from_trt_results(self, results: np.ndarray, image: Image):
        image_height, image_width, *_ = image.shape
        return [
            self._construct_object_from_trt_result(result, image_height, image_width)
            for result in results
            if TRTResultGetters.CLS.get_value(result) >= 0
        ]


class TRTResultGetters(Enum):
    CLS = (1, int)
    CONF = (2, float)
    X_MIN = (3, float)
    X_MAX = (5, float)
    Y_MIN = (4, float)
    Y_MAX = (6, float)

    def __init__(self, offset: int, type_: type):
        self.type_ = type_
        self.offset = offset

    def get_value(self, result: List[float]):
        return self.type_(result[self.offset])
