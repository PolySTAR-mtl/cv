from unittest import TestCase

from polystar.filters.filter_abc import FilterABC
from polystar.filters.keep_filter import KeepFilter


class OddFilter(FilterABC[int]):
    def validate_single(self, n: int) -> bool:
        return not n % 2


class TestFilterABC(TestCase):
    def test_filter(self):
        f = OddFilter()

        numbers = [1, 2, 3, 4, 5, 6]

        self.assertEqual([2, 4, 6], f.filter(numbers))

    def test_filter_with_siblings(self):
        f = OddFilter()

        numbers = [1, 2, 3, 4, 5, 6]
        names = list("abcdef")
        squares = [1, 4, 9, 16, 25, 36]

        f_numbers, f_names, f_squares = f.filter_with_siblings(numbers, names, squares)

        self.assertEqual([2, 4, 6], f_numbers)
        self.assertEqual(["b", "d", "f"], f_names)
        self.assertEqual([4, 16, 36], f_squares)

    def test_split(self):
        f = OddFilter()

        numbers = [1, 2, 3, 4, 5, 6]

        self.assertEqual(([1, 3, 5], [2, 4, 6]), f.split(numbers))

    def test_split_with_siblings(self):
        f = OddFilter()

        numbers = [1, 2, 3, 4, 5, 6]
        names = list("abcdef")
        squares = [1, 4, 9, 16, 25, 36]

        (
            (f_numbers_neg, f_names_neg, f_squares_neg),
            (f_numbers_pos, f_names_pos, f_squares_pos),
        ) = f.split_with_siblings(numbers, names, squares)

        self.assertEqual([2, 4, 6], f_numbers_pos)
        self.assertEqual(["b", "d", "f"], f_names_pos)
        self.assertEqual([4, 16, 36], f_squares_pos)
        self.assertEqual([1, 3, 5], f_numbers_neg)
        self.assertEqual(["a", "c", "e"], f_names_neg)
        self.assertEqual([1, 9, 25], f_squares_neg)

    def test_validate(self):
        f = OddFilter()

        numbers = [1, 2, 3, 4, 5, 6]

        self.assertEqual(([False, True, False, True, False, True]), f.validate(numbers))

    def test_or(self):
        f = KeepFilter([2, 3]) | KeepFilter([3, 4])

        numbers = [1, 2, 3, 4, 5, 6]

        self.assertEqual(([False, True, True, True, False, False]), f.validate(numbers))

    def test_and(self):
        f = KeepFilter([2, 3]) & KeepFilter([3, 4])

        numbers = [1, 2, 3, 4, 5, 6]

        self.assertEqual(([False, False, True, False, False, False]), f.validate(numbers))
