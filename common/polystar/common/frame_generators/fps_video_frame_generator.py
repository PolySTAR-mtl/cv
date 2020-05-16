from typing import Iterable

import ffmpeg
from dataclasses import dataclass

from polystar.common.frame_generators.video_frame_generator import VideoFrameGenerator
from polystar.common.models.image import Image


@dataclass
class FPSVideoFrameGenerator(VideoFrameGenerator):

    desired_fps: int

    def __post_init__(self):
        self.frame_rate: int = self._get_video_fps() // self.desired_fps

    def _get_video_fps(self):
        return max(
            int(stream["r_frame_rate"].split("/")[0]) for stream in ffmpeg.probe(str(self.video_path))["streams"]
        )

    def generate(self) -> Iterable[Image]:
        for i, frame in enumerate(super().generate()):
            if not i % self.frame_rate:
                yield frame
