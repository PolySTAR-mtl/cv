import logging
import pickle
from pathlib import Path
from typing import Any, Type

from polystar.common.utils.registry import registry


class UnpicklerWithRegistry(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Type:
        try:
            return registry[name]
        except KeyError:
            return super().find_class(module, name)


def pkl_load(file_path: Path):
    with file_path.with_suffix(".pkl").open("rb") as f:
        return UnpicklerWithRegistry(f).load()


def pkl_dump(obj: Any, file_path: Path):
    file_path_with_suffix = file_path.with_suffix(".pkl")
    file_path_with_suffix.write_bytes(pickle.dumps(obj))
    logging.info(f"{obj} saved at {file_path_with_suffix}")
