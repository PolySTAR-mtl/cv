from injector import inject

from polystar.common.dependency_injection import make_injector
from polystar.common.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.common.target_pipeline.objects_filters.armor_digit_filter import KeepArmorsDigitFilter
from polystar.common.target_pipeline.target_pipeline import NoTargetFoundException
from polystar.common.view.plt_results_viewer import PltResultViewer
from research.common.datasets.roco.roco_annotation_filters.roco_annotation_object_filter import (
    ROCOAnnotationObjectFilter,
)
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


@inject
def demo_pipeline_on_images(pipeline: DebugTargetPipeline):
    with PltResultViewer("Demo of tf model") as viewer:
        for builder in ROCODatasetsZoo.DEFAULT_TEST_DATASETS:
            for image in (
                builder.to_images()
                .filter_targets(ROCOAnnotationObjectFilter(KeepArmorsDigitFilter((1, 3, 4))))
                .shuffle()
                .cap(15)
                .build_examples()
            ):
                try:
                    pipeline.predict_target(image)
                except NoTargetFoundException:
                    pass
                viewer.display_debug_info(pipeline.debug_info_)


if __name__ == "__main__":
    make_injector().call_with_injection(demo_pipeline_on_images)
