from dataclasses import dataclass
from pathlib import Path
from typing import List

from more_itertools import ilen

from polystar.utils.iterable_utils import chunk
from polystar.utils.path import archive_directory, move_files
from research.common.constants import TWITCH_DSET_DIR, TWITCH_ROBOTS_VIEWS_DIR


@dataclass
class ImagesChunker:

    dataset_dir: Path
    chunks_dir: Path

    chunk_size: int = 100

    def make_chunks(self):
        images = self.dataset_dir.glob("**/*.jpg")
        for chunk_images in chunk(images, self.chunk_size):
            self._make_chunk(chunk_images)

    def _make_chunk(self, images: List[Path]):
        chunk_dir = self._get_next_available_chunk()
        move_files(images, chunk_dir)
        archive_directory(chunk_dir)

    def _get_next_available_chunk(self) -> Path:
        chunk_number = ilen(self.chunks_dir.glob("chunk_*.zip"))
        return self.chunks_dir / f"chunk_{chunk_number:03}"


if __name__ == "__main__":
    ImagesChunker(TWITCH_ROBOTS_VIEWS_DIR, TWITCH_DSET_DIR / "chunks-to-annotate").make_chunks()
