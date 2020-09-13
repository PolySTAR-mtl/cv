from unittest import TestCase
from unittest.mock import MagicMock

from research.common.datasets.dataset import Dataset, LazyDataset
from research.common.datasets.simple_dataset import SimpleDataset


class TestDataset(TestCase):
    def test_transform(self):
        dataset = _make_fake_dataset()

        str_str_dataset: Dataset[str, str] = dataset.transform(str, str)

        self.assertEqual([("0", "8"), ("1", "9"), ("2", "10"), ("3", "11")], list(str_str_dataset))

    def test_transform_examples(self):
        dataset = _make_fake_dataset()

        str_int_dataset: Dataset[str, int] = dataset.transform_examples(str)

        self.assertEqual([("0", 8), ("1", 9), ("2", 10), ("3", 11)], list(str_int_dataset))

    def test_transform_not_exhaustible(self):
        dataset = _make_fake_dataset()

        str_int_dataset: Dataset[str, float] = dataset.transform_examples(str)

        self.assertEqual([("0", 8), ("1", 9), ("2", 10), ("3", 11)], list(str_int_dataset))
        self.assertEqual([("0", 8), ("1", 9), ("2", 10), ("3", 11)], list(str_int_dataset))
        self.assertEqual([("0", 8), ("1", 9), ("2", 10), ("3", 11)], list(str_int_dataset))


class TestSimpleDataset(TestCase):
    def test_properties(self):
        dataset = _make_fake_dataset()

        self.assertEqual([0, 1, 2, 3], dataset.examples)
        self.assertEqual([8, 9, 10, 11], dataset.targets)

    def test_iter(self):
        dataset = _make_fake_dataset()

        self.assertEqual([(0, 8), (1, 9), (2, 10), (3, 11)], list(dataset))

    def test_len(self):
        dataset = _make_fake_dataset()

        self.assertEqual(4, len(dataset))

    def test_consistency(self):
        with self.assertRaises(AssertionError):
            SimpleDataset([0, 1], [8, 9, 10, 11], "fake")


class FakeLazyDataset(LazyDataset):
    def __init__(self):
        super().__init__("fake")

    __iter__ = MagicMock(side_effect=lambda *args: iter([(1, 1), (2, 4), (3, 9)]))


class TestLazyDataset(TestCase):
    def test_properties(self):
        dataset = FakeLazyDataset()

        self.assertEqual([1, 2, 3], list(dataset.examples))
        self.assertEqual([1, 4, 9], list(dataset.targets))
        self.assertEqual([(1, 1), (2, 4), (3, 9)], list(zip(dataset.examples, dataset.targets)))

    def test_properties_laziness(self):
        FakeLazyDataset.__iter__.reset_mock()
        dataset = FakeLazyDataset()

        list(dataset.examples)
        list(dataset.targets)
        FakeLazyDataset.__iter__.assert_called_once()

        FakeLazyDataset.__iter__.reset_mock()

        list(zip(dataset.examples, dataset.targets))
        FakeLazyDataset.__iter__.assert_called_once()


def _make_fake_dataset() -> Dataset[int, int]:
    return SimpleDataset([0, 1, 2, 3], [8, 9, 10, 11], "fake")
