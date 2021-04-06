from polystar.filters.filter_abc import FilterABC
from polystar.target_pipeline.objects_filters.in_box_filter import InBoxObjectFilter
from polystar.target_pipeline.objects_filters.type_object_filter import ARMORS_FILTER
from polystar.view.plt_results_viewer import PltResultViewer
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.roco_detection.small_base_filter import SMALL_BASE_FILTER


def clear_small_bases(annotation: ROCOAnnotation):
    small_bases, annotation.objects = SMALL_BASE_FILTER.split(annotation.objects)

    if not small_bases:
        return

    armors, robots = ARMORS_FILTER.split(annotation.objects)
    for base in small_bases:
        armors = (-InBoxObjectFilter(base.box, 0.5)).filter(armors)
    annotation.objects = robots + armors


class AnnotationHasObjectsFilter(FilterABC[ROCOAnnotation]):
    def validate_single(self, annotation: ROCOAnnotation) -> bool:
        return bool(annotation.objects)


if __name__ == "__main__":
    for _img, _annotation, _name in (
        (ROCODatasetsZoo.TWITCH.T470149066 | ROCODatasetsZoo.TWITCH.T470149568)
        .shuffle()
        .cap(10)
        .to_air()
        .filter_targets(AnnotationHasObjectsFilter())
        .cap(30)
    ):
        with PltResultViewer(_name) as _viewer:
            _viewer.display_image_with_objects(_img, _annotation.objects)
