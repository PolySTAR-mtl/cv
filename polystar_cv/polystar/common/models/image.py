from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from polystar.common.constants import PROJECT_DIR

Image = np.ndarray


@dataclass
class FileImage:
    path: Path
    image: Image = field(default=None)

    def __post_init__(self):
        if not self.image:
            self.image = load_image(self.path)

    def __array__(self) -> np.ndarray:
        return self.image

    def __getstate__(self) -> str:
        return str(self.path.relative_to(PROJECT_DIR))

    def __setstate__(self, rel_path: str):
        self.path = PROJECT_DIR / rel_path
        self.image = load_image(self.path)


def load_image(image_path: Path, conversion: int = cv2.COLOR_BGR2RGB) -> Image:
    return cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED), conversion)


def load_images(images: Iterable[Path], conversion: int = cv2.COLOR_BGR2RGB) -> Iterable[Image]:
    return (load_image(p, conversion) for p in images)


def load_images_in_directory(
    directory: Path, pattern: str = "*", conversion: int = cv2.COLOR_BGR2RGB
) -> Iterable[Image]:
    return load_images(directory.glob(pattern), conversion)


def save_image(image: Image, image_path: Path, conversion: int = cv2.COLOR_RGB2BGR):
    image_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(image, conversion))


def file_images_to_images(file_images: Iterable[FileImage]) -> List[Image]:
    return [np.asarray(file_image) for file_image in file_images]
