from pathlib import Path

import cv2
from nptyping import Array
from skimage import io


class Image(Array[int, ..., ..., 3]):
    @staticmethod
    def from_path(image_path: Path) -> "Image":
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
