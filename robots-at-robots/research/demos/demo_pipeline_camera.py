from time import time

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from polystar.common.constants import MODELS_DIR
from polystar.common.frame_generators.camera_frame_generator import CameraFrameGenerator
from polystar.common.models.label_map import LabelMap
from polystar.common.models.trt_model import TRTModel
from polystar.common.pipeline.objects_detectors.trt_model_object_detector import TRTModelObjectsDetector
from polystar.common.pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.blend_object_on_image import blend_object_on_image
from polystar.common.view.blend_text_on_image import blend_boxed_text_on_image
from polystar.robots_at_robots.dependency_injection import make_injector
from polystar.robots_at_robots.globals import settings

WINDOWS_NAME = "TensorRT demo"


[pycuda.autoinit]  # So pycharm won't remove the import


if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    objects_detector = TRTModelObjectsDetector(
        TRTModel(MODELS_DIR / settings.MODEL_NAME, (300, 300)), injector.get(LabelMap)
    )
    filters = [ConfidenceObjectValidator(confidence_threshold=0.5)]

    fps = 0
    try:
        for image in CameraFrameGenerator(1_280, 720).generate():
            previous_time = time()
            objects = objects_detector.detect(image)
            for f in filters:
                objects = f.filter(objects, image)

            fps = 0.9 * fps + 0.1 / (time() - previous_time)
            blend_boxed_text_on_image(image, f"FPS: {fps:.1f}", (10, 10), (0, 0, 0))

            for obj in objects:
                blend_object_on_image(image, obj)

            # Display the resulting frame
            cv2.imshow("frame", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # When everything done, release the capture
        cv2.destroyAllWindows()
