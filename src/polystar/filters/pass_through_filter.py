from typing import Any

from polystar.filters.filter_abc import FilterABC


class PassThroughFilter(FilterABC[Any]):
    def validate_single(self, example: Any) -> bool:
        return True
