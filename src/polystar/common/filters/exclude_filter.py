from typing import Iterable

from polystar.common.filters.filter_abc import FilterABC, T


class ExcludeFilter(FilterABC[T]):
    def __init__(self, to_remove: Iterable[T]):
        self.to_remove = set(to_remove)

    def validate_single(self, example: T) -> bool:
        return example not in self.to_remove
