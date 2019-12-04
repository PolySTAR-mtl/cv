from os import remove
from pathlib import Path
from shutil import move

import numpy as np
from scipy.spatial import distance
from skimage import io
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from research.constants import TWITCH_DSET


ref_image = io.imread('mask.jpg')

_MASK = ref_image[:, :, 1] > 50
_REF_IMG_MASKED = ref_image*_MASK[:, :, np.newaxis]
_THRESHOLD = 23


def is_image_from_robot_view(path_to_image: Path) -> bool:
    img = io.imread(path_to_image)
    img_masked = img * _MASK[:, :, np.newaxis]
    return distance.euclidean(img_masked.flatten() / 255, _REF_IMG_MASKED.flatten() / 255) < _THRESHOLD


class NewFrameHandler(FileSystemEventHandler):

    def __init__(self):
        self.res_dir = (TWITCH_DSET / 'robots-views')
        self.res_dir.mkdir(exist_ok=True, parents=True)

    def on_created(self, event: FileSystemEvent):
        if event.is_directory or not event.src_path.endswith('jpg'):
            return
        file_path = Path(event.src_path)
        if is_image_from_robot_view(file_path):
            res = self.res_dir / f'{file_path.parent.name}-{file_path.name}'
            move(file_path, res)
        else:
            remove(file_path)

    def __hash__(self):
        return hash(self.__class__.__name__)


if __name__ == '__main__':
    obs = Observer()
    obs.schedule(NewFrameHandler(), str(TWITCH_DSET / 'raw-frames'), recursive=True)
    obs.start()

    while 1:
        pass
