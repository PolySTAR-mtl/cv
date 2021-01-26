from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from polystar.frame_generators.cv2_frame_generator_abc import CV2FrameGeneratorABC


@dataclass
class VideoFrameGenerator(CV2FrameGeneratorABC):

    video_path: Path

    def _capture_params(self) -> Iterable[Any]:
        return (str(self.video_path),)
