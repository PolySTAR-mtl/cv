from typing import Iterable

from polystar.common.filters.filter_abc import FilterABC
from polystar.common.models.object import Armor, Object


class KeepArmorsDigitFilter(FilterABC[Object]):
    def __init__(self, digits: Iterable[int]):
        self.digits = digits

    def validate_single(self, obj: Object) -> bool:
        return isinstance(obj, Armor) and obj.number in self.digits
