import subprocess
import sys

import cv2

import pycuda.autoinit  # This is needed for initializing CUDA driver

from polystar.common.constants import MODELS_DIR
from polystar.common.models.label_map import LabelMap
from polystar.common.models.trt_model import TRTModel
from polystar.common.pipeline.objects_detectors.trt_model_object_detector import TRTModelObjectsDetector
from polystar.common.pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.bend_object_on_image import bend_object_on_image
from polystar.robots_at_robots.dependency_injection import make_injector
from polystar.robots_at_robots.globals import settings

WINDOWS_NAME = "TensorRT demo"


[pycuda.autoinit]  # So pycharm won't remove the import


def open_cam_onboard(width, height):
    """Open the Jetson onboard camera."""
    gst_elements = str(subprocess.check_output("gst-inspect-1.0"))
    if "nvcamerasrc" in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = (
            "nvcamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)2592, height=(int)1458, "
            "format=(string)I420, framerate=(fraction)30/1 ! "
            "nvvidconv ! "
            "video/x-raw, width=(int){}, height=(int){}, "
            "format=(string)BGRx ! "
            "videoconvert ! appsink"
        ).format(width, height)
    elif "nvarguscamerasrc" in gst_elements:
        gst_str = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)1920, height=(int)1080, "
            "format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=2 ! "
            "video/x-raw, width=(int){}, height=(int){}, "
            "format=(string)BGRx ! "
            "videoconvert ! appsink"
        ).format(width, height)
    else:
        raise RuntimeError("onboard camera source not found!")
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    objects_detector = TRTModelObjectsDetector(
        TRTModel(MODELS_DIR / settings.MODEL_NAME, (300, 300)), injector.get(LabelMap)
    )
    filters = [ConfidenceObjectValidator(confidence_threshold=0.5)]

    cap = open_cam_onboard(1_280, 720)

    if not cap.isOpened():
        sys.exit("Failed to open camera!")

    while True:
        ret, image = cap.read()
        objects = objects_detector.detect(image)
        for f in filters:
            objects = f.filter(objects, image)

        for obj in objects:
            bend_object_on_image(image, obj)

        # Display the resulting frame
        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
