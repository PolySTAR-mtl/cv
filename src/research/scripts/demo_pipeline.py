from injector import inject

from polystar.dependency_injection import make_injector
from polystar.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.target_pipeline.objects_filters.armor_digit_filter import KeepArmorsDigitFilter
from polystar.view.plt_results_viewer import PltResultViewer
from research.common.datasets.roco.roco_annotation_filters.roco_annotation_object_filter import (
    ROCOAnnotationObjectFilter,
)
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


@inject
def demo_pipeline_on_images(pipeline: DebugTargetPipeline):
    with PltResultViewer("Demo of tf model") as viewer:
        for builder in ROCODatasetsZoo.DEFAULT_TEST_DATASETS:
            for debug_info in pipeline.flow_debug(
                builder.to_images()
                .filter_targets(ROCOAnnotationObjectFilter(KeepArmorsDigitFilter((1, 3, 4))))
                .shuffle()
                .cap(15)
                .build_examples()
            ):
                viewer.display_debug_info(debug_info)


if __name__ == "__main__":
    make_injector().call_with_injection(demo_pipeline_on_images)
