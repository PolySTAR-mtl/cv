from cmath import pi
from unittest import TestCase

from numpy import empty

from polystar.dependency_injection import make_injector
from polystar.models.box import Box
from polystar.models.camera import Camera
from polystar.models.roco_object import ObjectType
from polystar.target_pipeline.detected_objects.detected_object import DetectedROCOObject
from polystar.target_pipeline.target_abc import SimpleTarget
from polystar.target_pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.target_pipeline.target_factories.target_factory_abc import TargetFactoryABC


class TestRatioTargetFactory(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        injector = make_injector()
        cls.target_factory: RatioSimpleTargetFactory = injector.get(TargetFactoryABC)
        cls.camera = injector.get(Camera)
        cls.h, cls.w = cls.camera.vertical_resolution, cls.camera.horizontal_resolution

    def test_right_ahead(self):
        self.assert_correct_angles(center_x=self.w // 2, phi=0, center_y=self.h // 2, theta=pi / 2)

    def test_full_up(self):
        self.assert_correct_vertical_angle(center_y=0, theta=pi / 2 - self.camera.vertical_fov / 2)

    def test_full_down(self):
        self.assert_correct_vertical_angle(center_y=self.h, theta=pi / 2 + self.camera.vertical_fov / 2)

    def test_full_left(self):
        self.assert_correct_horizontal_angle(center_x=0, phi=self.camera.horizontal_fov / 2)

    def test_full_right(self):
        self.assert_correct_horizontal_angle(center_x=self.w, phi=-self.camera.horizontal_fov / 2)

    # HELPERS

    def assert_correct_vertical_angle(self, *, center_y: int, theta: float):
        self.assert_correct_angles(center_y=center_y, theta=theta, center_x=self.w // 2, phi=0)

    def assert_correct_horizontal_angle(self, *, center_x: int, phi: float):
        self.assert_correct_angles(center_x=center_x, phi=phi, center_y=self.h // 2, theta=pi / 2)

    def assert_correct_angles(
        self, *, center_x: int, center_y: int, w: int = 10, h: int = 10, phi: float, theta: float
    ):
        obj = DetectedROCOObject(
            ObjectType.ARMOR, Box.from_size(center_x - w // 2, center_y + h // 2, w, h), confidence=1
        )
        img = empty((self.h, self.w, 3))
        target: SimpleTarget = self.target_factory.from_object(obj, img)
        self.assertAlmostEqual(theta, target.theta)
        self.assertAlmostEqual(phi, target.phi)
