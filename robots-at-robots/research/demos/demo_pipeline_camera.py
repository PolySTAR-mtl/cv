import subprocess

import cv2

from polystar.common.models.camera import Camera
from polystar.common.models.object import ObjectType
from polystar.common.pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.pipeline.objects_validators.type_object_validator import TypeObjectValidator
from polystar.common.pipeline.pipeline import Pipeline
from polystar.common.pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.robots_at_robots.dependency_injection import make_injector
from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset


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

    # objects_detector = injector.get(TFModelObjectsDetector)
    # filters = [ConfidenceObjectValidator(confidence_threshold=0.5)]

    cap = open_cam_onboard(0)

    for i, image_path in enumerate(SplitDataset(DJIROCODataset.CentralChina, Split.Test).image_paths):
        ret, image = cap.read()
        # objects = objects_detector.detect(image)

        # Display the resulting frame
        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
