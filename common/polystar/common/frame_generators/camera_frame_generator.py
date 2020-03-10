from dataclasses import dataclass
from typing import Any, Iterable

import cv2

from polystar.common.frame_generators.cv2_frame_generator_abc import CV2FrameGeneratorABC


@dataclass
class CameraFrameGenerator(CV2FrameGeneratorABC):
    width: int
    height: int

    def _capture_params(self) -> Iterable[Any]:
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            f"width=(int){self.width}, height=(int){self.height}, "
            "format=(string)NV12, framerate=(fraction)60/1 ! "
            "nvvidconv flip-method=0 ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, "
            "format=(string)BGRx ! "
            "videoconvert ! appsink",
            cv2.CAP_GSTREAMER,
        )
