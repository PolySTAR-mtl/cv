from typing import List

import numpy as np
from dataclasses import dataclass, field

from polystar.common.models.image import Image
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.target_abc import TargetABC
from polystar.common.target_pipeline.target_pipeline import TargetPipeline


@dataclass
class DebugInfo:
    detected_objects: List[DetectedObject] = field(init=False)
    validated_objects: List[DetectedObject] = field(init=False)
    selected_object: DetectedObject = field(init=False)
    target: TargetABC = field(init=False)


@dataclass
class DebugTargetPipeline(TargetPipeline):
    """Wrap a pipeline with debug, to store debug infos"""

    debug_info_: DebugInfo = field(init=False, default_factory=DebugInfo)

    def predict_target(self, image: Image) -> TargetABC:
        self.debug_info_ = DebugInfo()
        target = super().predict_target(image)
        self.debug_info_.target = target
        return target

    def predict_best_object(self, image: Image) -> DetectedObject:
        best_object = super().predict_best_object(image)
        self.debug_info_.selected_object = best_object
        return best_object

    def _get_objects_of_interest(self, image: np.ndarray) -> List[DetectedObject]:
        objects = super()._get_objects_of_interest(image)
        self.debug_info_.validated_objects = objects
        return objects

    def _detect_all_objects(self, image) -> List[DetectedObject]:
        objects = super()._detect_all_objects(image)
        self.debug_info_.detected_objects = objects
        return objects
