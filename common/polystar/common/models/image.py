from pathlib import Path
from typing import Iterable

import cv2
from nptyping import Array


class Image(Array[int, ..., ..., 3]):
    @staticmethod
    def from_path(image_path: Path, conversion: int = cv2.COLOR_BGR2RGB) -> "Image":
        return cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED), conversion)

    def save(self, image_path: Path, conversion: int = cv2.COLOR_RGB2BGR):
        image_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(image_path), cv2.cvtColor(self, conversion))


def load_images_in_directory(
    directory: Path, pattern: str = "*", conversion: int = cv2.COLOR_BGR2RGB
) -> Iterable[Image]:
    for image_path in directory.glob(pattern):
        yield Image.from_path(image_path, conversion)
