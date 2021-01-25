import unittest

from polystar.common.models.box import Box
from polystar.common.models.object import Object, ObjectType
from polystar.common.target_pipeline.objects_filters.in_box_filter import InBoxObjectFilter


class TestInBoxObjectFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.in_box_validator = InBoxObjectFilter(Box.from_size(2, 2, 6, 4), 0.5)

    def test_fully_inside(self):
        self._test_obj(3, 3, 2, 2, True)

    def test_on_left(self):
        self._test_obj(0, 3, 2, 2, False)

    def test_on_right(self):
        self._test_obj(9, 3, 2, 2, False)

    def test_on_top(self):
        self._test_obj(3, 0, 2, 2, False)

    def test_on_bottom(self):
        self._test_obj(3, 9, 2, 2, False)

    def test_on_diag(self):
        self._test_obj(9, 9, 2, 2, False)

    def test_half_left(self):
        self._test_obj(1, 3, 2, 2, True)

    def test_half_bottom(self):
        self._test_obj(3, 5, 2, 2, True)

    def test_half_corner_inside(self):
        self._test_obj(1, 1, 4, 4, True)

    def test_half_corner_outside(self):
        self._test_obj(1, 1, 3, 3, False)

    def test_two_third_top(self):
        self._test_obj(3, 0, 3, 3, False)

    def _test_obj(self, x: int, y: int, w: int, h: int, is_inside: bool):
        self.assertEqual(
            is_inside, self.in_box_validator.validate_single(Object(ObjectType.CAR, Box.from_size(x, y, w, h)))
        )
