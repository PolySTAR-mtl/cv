from dataclasses import dataclass

from polystar.common.models.box import Box
from polystar.common.models.object import Object
from polystar.common.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC


@dataclass
class InBoxObjectFilter(ObjectsFilterABC):
    box: Box
    min_percentage_intersection: float

    def validate_single(self, obj: Object) -> bool:
        return self.box.contains(obj.box, self.min_percentage_intersection)
