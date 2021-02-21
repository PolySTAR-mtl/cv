from dataclasses import dataclass
from typing import Iterable

from polystar.frame_generators.video_frame_generator import VideoFrameGenerator
from polystar.models.image import Image


@dataclass
class FPSVideoFrameGenerator(VideoFrameGenerator):

    desired_fps: int

    def __post_init__(self):
        self.frame_rate: int = self._video_fps // self.desired_fps

    def generate(self) -> Iterable[Image]:
        for i, frame in enumerate(super().generate(), -1):
            if not i % self.frame_rate:
                yield frame
