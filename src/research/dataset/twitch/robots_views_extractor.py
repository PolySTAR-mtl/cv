import cv2
import ffmpeg
import numpy as np
from tqdm import tqdm

from polystar.frame_generators.fps_video_frame_generator import FPSVideoFrameGenerator
from research.constants import TWITCH_DSET_DIR, TWITCH_ROBOTS_VIEWS_DIR
from research.dataset.twitch.mask_detector import has_bonus_icon, robot_view_mask_hd


class RobotsViewExtractor:

    FPS = 2
    OFFSET_SECONDS = 3140 // 2

    def __init__(self, video_name: str):
        self.video_name: str = video_name
        self.video_path = TWITCH_DSET_DIR / "videos" / f"{video_name}.mp4"
        self.frame_generator: FPSVideoFrameGenerator = FPSVideoFrameGenerator(
            self.video_path, self.OFFSET_SECONDS, self.FPS
        )
        self.count = 0
        (TWITCH_ROBOTS_VIEWS_DIR / self.video_name).mkdir(exist_ok=True)
        self._progress_bar = None

    def run(self):
        self._progress_bar = tqdm(
            enumerate(self.frame_generator, 1 + self.OFFSET_SECONDS * self.FPS),
            total=self._get_number_of_frames(),
            desc=f"Extracting robots views from video {self.video_name}.mp4",
            unit="frames",
            ncols=200,
        )
        for i, frame in self._progress_bar:
            self._process_frame(frame, i)
        print(f"Detected {self.count} robots views")

    def _process_frame(self, frame: np.ndarray, frame_number: int):
        if robot_view_mask_hd.match(frame) and not has_bonus_icon(frame):
            self._save_frame(frame, frame_number)
            self.count += 1
            self._progress_bar.set_description(
                f"Extracting robots views from video {self.video_name}.mp4 ({self.count} so far)"
            )

    def _save_frame(self, frame: np.ndarray, frame_number: int):
        cv2.imwrite(f"{TWITCH_ROBOTS_VIEWS_DIR}/{self.video_name}/{self.video_name}-frame-{frame_number:06}.jpg", frame)

    def _get_number_of_frames(self):
        return int(ffmpeg.probe(str(self.video_path))["format"]["duration"].split(".")[0]) * self.FPS
