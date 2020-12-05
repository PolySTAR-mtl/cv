import cv2

from polystar.common.communication.print_target_sender import PrintTargetSender
from polystar.common.models.camera import Camera
from polystar.common.models.label_map import LabelMap
from polystar.common.target_pipeline.armors_descriptors.armors_color_descriptor import ArmorsColorDescriptor
from polystar.common.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.common.target_pipeline.detected_objects.detected_objects_factory import DetectedObjectFactory
from polystar.common.target_pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.target_pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.target_pipeline.objects_linker.simple_objects_linker import SimpleObjectsLinker
from polystar.common.target_pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.target_pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.common.target_pipeline.target_pipeline import NoTargetFoundException
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.plt_results_viewer import PltResultViewer
from polystar.robots_at_robots.dependency_injection import make_injector
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.armor_color.baseline_experiments import (
    ArmorColorPipeline,
    MeanChannels,
    RedBlueComparisonClassifier,
)
from research.robots_at_robots.demos.utils import load_tf_model

if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    pipeline = DebugTargetPipeline(
        objects_detector=TFModelObjectsDetector(
            DetectedObjectFactory(
                injector.get(LabelMap),
                [ArmorsColorDescriptor(ArmorColorPipeline.from_pipes([MeanChannels(), RedBlueComparisonClassifier()]))],
            ),
            load_tf_model(),
        ),
        objects_validators=[ConfidenceObjectValidator(0.6)],
        object_selector=ClosestObjectSelector(),
        target_factory=RatioSimpleTargetFactory(injector.get(Camera), 300, 100),
        target_sender=PrintTargetSender(),
        objects_linker=SimpleObjectsLinker(min_percentage_intersection=0.8),
    )

    with PltResultViewer("Demo of tf model") as viewer:
        for builder in (ROCODatasetsZoo.TWITCH.T470150052, ROCODatasetsZoo.DJI.CENTRAL_CHINA):
            for image_path, _, _ in builder.cap(5):
                try:
                    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                    target = pipeline.predict_target(image)
                except NoTargetFoundException:
                    pass
                finally:
                    viewer.display_debug_info(pipeline.debug_info_)
