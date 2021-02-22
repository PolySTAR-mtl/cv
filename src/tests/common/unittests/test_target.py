from unittest import TestCase

from polystar.target_pipeline.target_abc import SimpleTarget


class TestTarget(TestCase):
    def test_bytes(self):
        target = SimpleTarget(theta=0.25, phi=-0.35, d=5.3)

        self.assertEqual(b"\xfa\x00\xa2\xfe\xb4\x14", bytes(target))
