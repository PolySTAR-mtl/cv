from abc import ABC
from math import sin, asin, sqrt, atan2
from typing import Tuple

import numpy as np

from polystar.common.models.camera import Camera
from polystar.common.models.object import Object
from polystar.common.pipeline.target_factories.target_factory_abc import TargetFactoryABC


class RatioTargetFactoryABC(TargetFactoryABC, ABC):
    def __init__(self, camera: Camera, real_width: float, real_height: float):
        self._ratio_w = real_width * camera.w // 2 / sin(camera.horizontal_angle)
        # self._ratio_h = real_height * camera.h // 2 / sin(camera.vertical_angle)

        self._coef_angle = sin(camera.horizontal_angle) / (camera.w // 2)

    def _calculate_angles(self, obj: Object, image: np.ndarray) -> Tuple[float, float]:
        x, y = obj.x + obj.w // 2 - image.shape[1] // 2, image.shape[0] // 2 - obj.y + obj.h // 2
        phi = asin(sqrt(x ** 2 + y ** 2) * self._coef_angle)
        theta = atan2(y, x)
        return phi, theta

    def _calculate_distance(self, obj: Object) -> float:
        return self._ratio_w / obj.w
