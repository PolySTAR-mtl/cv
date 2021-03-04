from dataclasses import dataclass
from typing import Iterator

from polystar.frame_generators.video_frame_generator import VideoFrameGenerator
from polystar.models.image import Image


@dataclass
class FPSVideoFrameGenerator(VideoFrameGenerator):

    desired_fps: int

    def __post_init__(self):
        self.frame_rate: int = self._video_fps // self.desired_fps

    def __iter__(self) -> Iterator[Image]:
        for i, frame in enumerate(super().__iter__(), -1):
            if not i % self.frame_rate:
                yield frame
