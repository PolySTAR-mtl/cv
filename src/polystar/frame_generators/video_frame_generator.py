from pathlib import Path
from typing import Iterable, Iterator, Optional

import ffmpeg
from cv2.cv2 import CAP_PROP_POS_FRAMES
from memoized_property import memoized_property

from polystar.frame_generators.cv2_frame_generator_abc import CV2Capture, CV2FrameGenerator
from polystar.models.image import Image


class VideoFrameGenerator(CV2FrameGenerator):
    def __init__(self, video_path: Path, offset_seconds: Optional[int] = None):
        super().__init__(str(video_path))
        self.offset_seconds = offset_seconds
        self.video_path = video_path

    def __iter__(self) -> Iterator[Image]:
        if self.offset_seconds:
            return CV2CaptureWithOffset(self.capture_params, self._video_fps * self.offset_seconds - 2)
        return CV2Capture(self.capture_params)

    @memoized_property
    def _video_fps(self) -> int:
        streams_info = ffmpeg.probe(str(self.video_path))["streams"]
        for stream_info in streams_info:
            if stream_info["codec_type"] != "video":
                continue
            return round(eval(stream_info["avg_frame_rate"]))
        raise ValueError(f"No fps found for video {self.video_path.name}")


class CV2CaptureWithOffset(CV2Capture):
    def __init__(self, capture_params: Iterable, offset_frames: int):
        super().__init__(capture_params)
        self._cap.set(CAP_PROP_POS_FRAMES, offset_frames)
