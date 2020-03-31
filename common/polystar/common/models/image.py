from pathlib import Path

import cv2
from nptyping import Array


class Image(Array[int, ..., ..., 3]):
    @staticmethod
    def from_path(image_path: Path) -> "Image":
        return cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    def save(self, image_path: Path):
        image_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(image_path), cv2.cvtColor(self, cv2.COLOR_RGB2BGR))
