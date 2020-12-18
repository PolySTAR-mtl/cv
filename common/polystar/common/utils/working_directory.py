from contextlib import contextmanager
from os import chdir
from pathlib import Path


@contextmanager
def working_directory(path: Path):
    prev_cwd = Path.cwd()
    chdir(str(path))
    try:
        yield
    finally:
        chdir(prev_cwd)
