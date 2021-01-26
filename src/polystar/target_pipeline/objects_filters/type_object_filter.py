from typing import Iterable

from polystar.models.roco_object import ObjectType, ROCOObject
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC


class TypeObjectsFilter(ObjectsFilterABC):
    """Keep only the objects of a desired type"""

    def __init__(self, desired_types: Iterable[ObjectType]):
        self.desired_types = set(desired_types)

    def validate_single(self, obj: ROCOObject) -> bool:
        return obj.type in self.desired_types
