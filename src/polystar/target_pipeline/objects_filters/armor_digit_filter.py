from typing import Iterable

from polystar.filters.filter_abc import FilterABC
from polystar.models.roco_object import Armor, ROCOObject


class KeepArmorsDigitFilter(FilterABC[ROCOObject]):
    def __init__(self, digits: Iterable[int]):
        self.digits = digits

    def validate_single(self, obj: ROCOObject) -> bool:
        return isinstance(obj, Armor) and obj.number in self.digits
