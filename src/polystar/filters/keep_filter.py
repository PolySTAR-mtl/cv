from typing import Iterable

from polystar.filters.filter_abc import FilterABC, T


class KeepFilter(FilterABC[T]):
    def __init__(self, to_keep: Iterable[T]):
        self.to_keep = set(to_keep)

    def validate_single(self, example: T) -> bool:
        return example in self.to_keep
