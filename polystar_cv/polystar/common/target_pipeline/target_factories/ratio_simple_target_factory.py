import numpy as np

from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.target_abc import TargetABC, SimpleTarget
from polystar.common.target_pipeline.target_factories.ratio_target_factory_abc import RatioTargetFactoryABC


class RatioSimpleTargetFactory(RatioTargetFactoryABC):
    def from_object(self, obj: DetectedObject, image: np.array) -> TargetABC:
        return SimpleTarget(self._calculate_distance(obj), *self._calculate_angles(obj, image))