from unittest import TestCase

from research.common.datasets.image_dataset import ImageDataset


class TestImageDataset(TestCase):
    def test_iter(self):
        dataset = ImageDataset("test", list(range(5)), list(range(3, 8)))

        self.assertEqual([(0, 3), (1, 4), (2, 5), (3, 6), (4, 7)], list(dataset))

    def test_auto_load(self):
        class FakeDataset(ImageDataset):
            def __iter__(self):
                return [(0, 2), (1, 4)].__iter__()

        dataset = FakeDataset("test")

        self.assertEqual([0, 1], dataset.images)
        self.assertEqual([2, 4], dataset.targets)

    def test_assert(self):
        with self.assertRaises(AssertionError):
            ImageDataset("test")

    def test_assert_child(self):
        class FakeDataset(ImageDataset):
            pass

        with self.assertRaises(AssertionError):
            FakeDataset("test")
