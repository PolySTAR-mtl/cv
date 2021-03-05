import cv2

from polystar.frame_generators.cv2_frame_generator_abc import CV2FrameGenerator


def make_csi_camera_frame_generator(width: int, height: int) -> CV2FrameGenerator:
    return CV2FrameGenerator(
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        "format=(string)NV12, framerate=60/1 ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, "
        "format=(string)BGRx ! "
        "videoconvert ! appsink drop=true sync=false",
        cv2.CAP_GSTREAMER,
    )
