from abc import ABC
from math import atan2, pi, tan
from typing import Tuple

import numpy as np

from polystar.models.camera import Camera
from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject
from polystar.target_pipeline.target_factories.target_factory_abc import TargetFactoryABC


class RatioTargetFactoryABC(TargetFactoryABC, ABC):
    def __init__(self, camera: Camera, real_width_m: float, real_height_m: float):
        self._vertical_distance_coef = camera.focal_m * real_height_m / camera.pixel_size_m
        self._vertical_angle_distance = camera.vertical_resolution / (2 * tan(camera.vertical_fov))
        self._horizontal_angle_distance = camera.horizontal_resolution / (2 * tan(camera.horizontal_fov))

    def _calculate_angles(self, obj: DetectedROCOObject, image: np.ndarray) -> Tuple[float, float]:
        x, y = obj.box.x + obj.box.w // 2 - image.shape[1] // 2, image.shape[0] // 2 - obj.box.y + obj.box.h // 2
        phi = -atan2(x, self._horizontal_angle_distance)
        theta = pi / 2 - atan2(y, self._vertical_angle_distance)
        return phi, theta

    def _calculate_distance(self, obj: DetectedROCOObject) -> float:
        return self._vertical_distance_coef / obj.box.h
