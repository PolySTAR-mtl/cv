import numpy as np

from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject
from polystar.target_pipeline.target_abc import SimpleTarget, TargetABC
from polystar.target_pipeline.target_factories.ratio_target_factory_abc import RatioTargetFactoryABC


class RatioSimpleTargetFactory(RatioTargetFactoryABC):
    def from_object(self, obj: DetectedROCOObject, image: np.array) -> TargetABC:
        return SimpleTarget(self._calculate_distance(obj), *self._calculate_angles(obj, image))
