from pathlib import Path

import cv2
import ffmpeg


class VideoFrameGenerator:
    def __init__(self, video_path: Path, desired_fps: int):
        self.video_path: Path = video_path
        self.desired_fps: int = desired_fps
        self.video_fps: int = self._get_video_fps()

    def _get_video_fps(self):
        return max(
            int(stream["r_frame_rate"].split("/")[0]) for stream in ffmpeg.probe(str(self.video_path))["streams"]
        )

    def generate(self):
        video = cv2.VideoCapture(str(self.video_path))
        frame_rate = self.video_fps // self.desired_fps
        count = 0
        while 1:
            is_unfinished, frame = video.read()
            if not is_unfinished:
                video.release()
                return
            if not count % frame_rate:
                yield frame
            count += 1
