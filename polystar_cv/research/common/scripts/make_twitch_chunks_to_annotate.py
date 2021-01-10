import shutil
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Iterator

from research.common.constants import TWITCH_ROBOTS_VIEWS_DIR


@dataclass
class DatasetChunker:

    dataset_dir: Path

    chunk_size: int = 100

    def make_chunks(self):
        try:
            image_paths_iterator = self.dataset_dir.glob("*.jpg")
            for chunk_number in count(1 + self._get_number_existing_chunks()):
                self._make_next_chunk(chunk_number, image_paths_iterator)
        except StopIteration:
            self._zip_chunk(self._get_chunk_dir(self._get_number_existing_chunks() + 1))

    def _make_next_chunk(self, chunk_number: int, image_paths_iterator: Iterator[Path]):
        chunk_dir = self._get_chunk_dir(chunk_number)
        chunk_dir.mkdir()
        for _ in range(self.chunk_size):
            image_path = next(image_paths_iterator)
            shutil.move(str(image_path), str(chunk_dir / image_path.name))
        self._zip_chunk(chunk_dir)

    def _get_chunk_dir(self, chunk_number: int):
        return self.dataset_dir / f"chunk_{chunk_number:03}"

    def _get_number_existing_chunks(self):
        return int(str(max(self.dataset_dir.glob("chunk_*.zip"), default="chunk_000.zip"))[-7:-4])

    @staticmethod
    def _zip_chunk(chunk_dir: Path):
        shutil.make_archive(str(chunk_dir), "zip", str(chunk_dir))


if __name__ == "__main__":
    DatasetChunker(TWITCH_ROBOTS_VIEWS_DIR).make_chunks()
