from dataclasses import dataclass

from polystar.models.roco_object import ROCOObject
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC


@dataclass
class SmallObjectFilter(ObjectsFilterABC):
    min_size: int

    def validate_single(self, obj: ROCOObject) -> bool:
        return obj.box.area >= self.min_size
