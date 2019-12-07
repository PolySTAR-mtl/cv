import sys
from dataclasses import dataclass

import ffmpeg
from pyprind import ProgBar
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from research.constants import TWITCH_DSET


class ThumbnailsGenerator:

    FPS = 2

    def __init__(self, video_name: str):
        self.output_folder = TWITCH_DSET / 'raw-frames' / video_name
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.video_path = TWITCH_DSET / 'videos' / f'{video_name}.mp4'

    def run(self):
        frames = self._get_frame_number() * self.FPS
        self._create_progress_bar(frames)
        self._launch_creation(frames)

    def _launch_creation(self, frames):
        ffmpeg. \
            input(str(self.video_path)). \
            filter('fps', fps=self.FPS).\
            output(f"{self.output_folder}/frame_%d.jpg", frames=frames). \
            run(quiet=True)

    @dataclass
    class ProgressHandler(FileSystemEventHandler):
        bar: ProgBar

        def on_created(self, event: FileSystemEvent):
            self.bar.update()

        def __hash__(self):
            return hash(self.__class__.__name__)

    def _create_progress_bar(self, frames):
        bar = ProgBar(frames, title='Creating thumbnails', width=100, stream=sys.stdout, update_interval=True)
        obs = Observer()
        obs.schedule(self.ProgressHandler(bar), str(self.output_folder), recursive=True)
        obs.start()

    def _get_frame_number(self):
        return int(ffmpeg.probe(str(self.video_path))['format']['duration'].split('.')[0])
