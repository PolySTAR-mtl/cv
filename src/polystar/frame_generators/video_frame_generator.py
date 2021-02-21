from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import ffmpeg
from cv2.cv2 import CAP_PROP_POS_FRAMES
from memoized_property import memoized_property

from polystar.frame_generators.cv2_frame_generator_abc import CV2FrameGeneratorABC


@dataclass
class VideoFrameGenerator(CV2FrameGeneratorABC):

    video_path: Path
    offset_seconds: Optional[int]

    def _capture_params(self) -> Iterable[Any]:
        return (str(self.video_path),)

    def _post_opening_operation(self):
        if self.offset_seconds:
            self._cap.set(CAP_PROP_POS_FRAMES, self._video_fps * self.offset_seconds - 2)

    @memoized_property
    def _video_fps(self) -> int:
        streams_info = ffmpeg.probe(str(self.video_path))["streams"]
        for stream_info in streams_info:
            if stream_info["codec_type"] != "video":
                continue
            return round(eval(stream_info["avg_frame_rate"]))
        raise ValueError(f"No fps found for video {self.video_path.name}")
