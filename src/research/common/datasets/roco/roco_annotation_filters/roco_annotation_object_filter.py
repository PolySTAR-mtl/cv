from polystar.filters.filter_abc import FilterABC
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC
from research.common.datasets.roco.roco_annotation import ROCOAnnotation


class ROCOAnnotationObjectFilter(FilterABC):
    def __init__(self, object_filter: ObjectsFilterABC):
        self.object_filter = object_filter

    def validate_single(self, annotation: ROCOAnnotation) -> bool:
        return any(self.object_filter.validate(annotation.objects))
