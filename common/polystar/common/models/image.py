from __future__ import annotations

from pathlib import Path

from nptyping import Array
from skimage import io


class Image(Array[int, ..., ..., 3]):
    @staticmethod
    def from_path(image_path: Path) -> Image:
        return io.imread(str(image_path))
