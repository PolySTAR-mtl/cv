from os import PathLike, remove
from pathlib import Path
from shutil import copy, make_archive, move
from typing import Iterable


def move_file(source: Path, destination_directory: Path) -> Path:
    destination_directory.mkdir(exist_ok=True, parents=True)
    destination = destination_directory / source.name
    if destination.exists():
        remove(destination)
    return Path(move(str(source), str(destination_directory)))


def move_files(sources: Iterable[Path], destination_directory: Path):
    for source in sources:
        move_file(source, destination_directory)


def copy_file(source: Path, destination_directory: Path) -> Path:
    destination_directory.mkdir(exist_ok=True, parents=True)
    return Path(copy(str(source), str(destination_directory)))


def archive_directory(directory:Path):
    make_archive(str(directory), "zip", str(directory))


def make_path(p: PathLike) -> Path:
    p = Path(p)
    p.mkdir(exist_ok=True, parents=True)
    return p
