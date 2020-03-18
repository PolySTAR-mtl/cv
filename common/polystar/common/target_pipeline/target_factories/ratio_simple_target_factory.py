import numpy as np

from polystar.common.models.object import Object
from polystar.common.models.target_abc import TargetABC, SimpleTarget
from polystar.common.target_pipeline.target_factories.ratio_target_factory_abc import RatioTargetFactoryABC


class RatioSimpleTargetFactory(RatioTargetFactoryABC):
    def from_object(self, obj: Object, image: np.array) -> TargetABC:
        return SimpleTarget(self._calculate_distance(obj), *self._calculate_angles(obj, image))
