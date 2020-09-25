from typing import List

from polystar.common.filters.filter_abc import FilterABC, T


class UnionFilter(FilterABC[T]):
    def __init__(self, filters: List[FilterABC[T]]):
        self.filters = filters
        assert self.filters

    def validate_single(self, example: T) -> bool:
        return any(f.validate_single(example) for f in example)
