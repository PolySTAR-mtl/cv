from pathlib import Path

import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm

from research_common.constants import TWITCH_DSET_DIR
from research_common.dataset.twitch.mask_detector import is_image_from_robot_view
from research_common.dataset.twitch.video_frame_generator import VideoFrameGenerator

RES_DIR: Path = TWITCH_DSET_DIR / "robots-views"
RES_DIR.mkdir(parents=True, exist_ok=True)


class RobotsViewExtractor:

    FPS = 2

    def __init__(self, video_name: str):
        self.video_name: str = video_name
        self.video_path = TWITCH_DSET_DIR / "videos" / f"{video_name}.mp4"
        self.frame_generator: VideoFrameGenerator = VideoFrameGenerator(self.video_path, self.FPS)
        self.count = 0

    def run(self):
        for i, frame in tqdm(
            enumerate(self.frame_generator.generate()),
            total=self._get_number_of_frames(),
            desc=f"Extracting robots views from video {self.video_name}.mp4",
            unit="frames",
            ncols=200,
        ):
            self._process_frame(frame, i)
        print(f"Detected {self.count} robots views")

    def _process_frame(self, frame: np.ndarray, frame_number: int):
        if is_image_from_robot_view(frame):
            self._save_frame(frame, frame_number)
            self.count += 1

    def _save_frame(self, frame: np.ndarray, frame_number: int):
        cv2.imwrite(f"{RES_DIR}/{self.video_name}-frame-{frame_number + 1:06}.jpg", frame)

    def _get_number_of_frames(self):
        return int(ffmpeg.probe(str(self.video_path))["format"]["duration"].split(".")[0]) * self.FPS
