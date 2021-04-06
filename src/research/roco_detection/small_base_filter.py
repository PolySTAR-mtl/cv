from polystar.models.roco_object import ObjectType
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC
from polystar.target_pipeline.objects_filters.size_filter import SmallObjectFilter
from polystar.target_pipeline.objects_filters.type_object_filter import TypeObjectsFilter

SMALL_BASE_FILTER: ObjectsFilterABC = -TypeObjectsFilter({ObjectType.BASE}) | SmallObjectFilter(12_500)
