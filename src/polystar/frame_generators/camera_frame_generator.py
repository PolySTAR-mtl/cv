from dataclasses import dataclass
from typing import Any, Iterable

import cv2

from polystar.frame_generators.cv2_frame_generator_abc import CV2FrameGeneratorABC


@dataclass
class RaspiV2CameraFrameGenerator(CV2FrameGeneratorABC):
    width: int
    height: int

    def _capture_params(self) -> Iterable[Any]:
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            f"width=(int){self.width}, height=(int){self.height}, "
            "format=(string)NV12, framerate=60/1 ! "
            "nvvidconv flip-method=0 ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, "
            "format=(string)BGRx ! "
            "videoconvert ! appsink drop=true sync=false",
            cv2.CAP_GSTREAMER,
        )


@dataclass
class WebcamFrameGenerator(CV2FrameGeneratorABC):
    def _capture_params(self) -> Iterable[Any]:
        return (0,)
