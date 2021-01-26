from dataclasses import dataclass

from polystar.models.box import Box
from polystar.models.roco_object import ROCOObject
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC


@dataclass
class ContainsBoxObjectsFilter(ObjectsFilterABC):
    box: Box
    min_percentage_intersection: float

    def validate_single(self, obj: ROCOObject) -> bool:
        return obj.box.contains(self.box, self.min_percentage_intersection)
