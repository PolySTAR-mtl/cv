from pathlib import Path
from typing import Iterator, Optional

from polystar.frame_generators.video_frame_generator import VideoFrameGenerator
from polystar.models.image import Image


class FPSVideoFrameGenerator(VideoFrameGenerator):
    def __init__(self, video_path: Path, desired_fps: int, offset_seconds: Optional[int] = None):
        super().__init__(video_path, offset_seconds)
        self.frame_rate: int = self._video_fps // desired_fps

    def __iter__(self) -> Iterator[Image]:
        for i, frame in enumerate(super().__iter__(), -1):
            if not i % self.frame_rate:
                yield frame
