import sys
from time import time

import pycuda.autoinit  # This is needed for initializing CUDA driver

from polystar.common.communication.file_descriptor_target_sender import FileDescriptorTargetSender
from polystar.common.constants import MODELS_DIR
from polystar.common.dependency_injection import make_injector
from polystar.common.frame_generators.camera_frame_generator import CameraFrameGenerator
from polystar.common.models.camera import Camera
from polystar.common.models.label_map import LabelMap
from polystar.common.models.object import ObjectType
from polystar.common.models.trt_model import TRTModel
from polystar.common.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.common.target_pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.target_pipeline.objects_detectors.trt_model_object_detector import TRTModelObjectsDetector
from polystar.common.target_pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.target_pipeline.objects_validators.type_object_validator import TypeObjectValidator
from polystar.common.target_pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.cv2_results_viewer import CV2ResultViewer
from polystar.robots_at_robots.globals import settings

[pycuda.autoinit]  # So pycharm won't remove the import


if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    objects_detector = TRTModelObjectsDetector(
        TRTModel(MODELS_DIR / settings.MODEL_NAME, (300, 300)), injector.get(LabelMap)
    )
    pipeline = DebugTargetPipeline(
        objects_detector=objects_detector,
        objects_validators=[ConfidenceObjectValidator(0.6), TypeObjectValidator(ObjectType.Armor)],
        object_selector=ClosestObjectSelector(),
        target_factory=RatioSimpleTargetFactory(injector.get(Camera), 300, 100),
        target_sender=FileDescriptorTargetSender(int(sys.argv[1])),
    )

    fps = 0
    with CV2ResultViewer("TensorRT demo") as viewer:
        for image in CameraFrameGenerator(1_280, 720).generate():

            previous_time = time()

            # inference
            pipeline.predict_target(image)

            # display
            fps = 0.9 * fps + 0.1 / (time() - previous_time)
            viewer.new(image)
            viewer.add_objects(pipeline.debug_info_.validated_objects, forced_color=(0.6, 0.6, 0.6))
            viewer.add_object(pipeline.debug_info_.selected_object)
            viewer.add_text(f"FPS: {fps:.1f}", 10, 10, (0, 0, 0))
            viewer.display()

            if viewer.finished:
                break
