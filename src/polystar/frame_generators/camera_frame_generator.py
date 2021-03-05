from threading import Thread
from typing import Any, Iterator

import cv2

from polystar.frame_generators.cv2_frame_generator_abc import CV2FrameGenerator
from polystar.models.image import Image


class CameraFrameGenerator(CV2FrameGenerator):
    def __init__(self, *capture_params: Any):
        super().__init__(*capture_params)
        self.camera_thread = CameraThread(super().__iter__())

    def __iter__(self):
        self.camera_thread.start()
        while True:
            yield self.camera_thread.current_frame.copy()

    def __del__(self):
        self.camera_thread.stop()


class CameraThread(Thread):
    def __init__(self, it: Iterator[Image]):
        super().__init__()
        self.it = it
        self.running = True
        self._get_next_frame()

    def run(self):
        while self.running:
            self._get_next_frame()

    def stop(self):
        self.running = False

    def _get_next_frame(self):
        try:
            self.current_frame = next(self.it)
        except StopIteration:
            self.running = False


def make_csi_camera_frame_generator(width: int, height: int) -> CameraFrameGenerator:
    return CameraFrameGenerator(
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
