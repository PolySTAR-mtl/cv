from typing import Iterable

from polystar.common.models.object import Object, ObjectType
from polystar.common.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC


class TypeObjectsFilter(ObjectsFilterABC):
    """Keep only the objects of a desired type"""

    def __init__(self, desired_types: Iterable[ObjectType]):
        self.desired_types = set(desired_types)

    def validate_single(self, obj: Object) -> bool:
        return obj.type in self.desired_types
