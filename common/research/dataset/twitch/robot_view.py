from os import remove
from pathlib import Path
from shutil import move

import numpy as np
from scipy.spatial import distance
from skimage import io
from skimage.transform import resize

from research.constants import TWITCH_DSET

RES_DIR: Path = TWITCH_DSET / 'robots-views'
RES_DIR.mkdir(parents=True, exist_ok=True)

ref_image = io.imread(f'{__file__}/../mask.jpg')

_MASK = ref_image[:, :, 1] > 50
_REF_IMG_MASKED = ref_image*_MASK[:, :, np.newaxis]
_THRESHOLD = 23


def is_image_from_robot_view(path_to_image: Path) -> bool:
    img = io.imread(path_to_image)
    img_masked = img * _MASK[:, :, np.newaxis]
    return distance.euclidean(img_masked.flatten() / 255, _REF_IMG_MASKED.flatten() / 255) < _THRESHOLD


def process_image(path_to_image: Path):
    if is_image_from_robot_view(path_to_image):
        res_path = RES_DIR / f'{path_to_image.parent.name}-{path_to_image.name}'
        move(str(path_to_image), str(res_path))
        print(f'{path_to_image.stem} is a robot view, moving it')
    else:
        remove(str(path_to_image))


def process_all_images_in_dir(dir_path: Path):
    for path_to_image in dir_path.glob('*/*.jpg'):
        process_image(path_to_image)
