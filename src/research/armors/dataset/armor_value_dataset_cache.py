import json
from pathlib import Path
from shutil import rmtree
from typing import ClassVar, Generic, Optional, TypeVar

from polystar.models.image import Image, save_image
from polystar.utils.time import create_time_id
from polystar.utils.tqdm import smart_tqdm
from research.common.datasets.lazy_dataset import LazyDataset

T = TypeVar("T")


# TODO: add AWS support here
class DatasetCache(Generic[T]):
    VERSION: ClassVar[str] = "2.0"

    def __init__(self, cache_dir: Path, dataset: LazyDataset[Image, T]):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.lock_file = cache_dir / ".lock"

        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def generate_if_missing(self):
        cause = self._get_generation_cause()
        if cause is None:
            return
        self._clean_cache_dir()
        self._generate(cause)

    def _get_generation_cause(self) -> Optional[str]:
        if not self.lock_file.exists():
            return "lock not found"
        version = json.loads(self.lock_file.read_text())["version"]
        if version != self.VERSION:
            return f"upgrade [{version} -> {self.VERSION}]"

    def _clean_cache_dir(self):
        rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir()

    def _generate(self, cause: str):
        desc = f"Generating dataset {self.dataset.name} (cause: {cause})"
        for img, target, name in smart_tqdm(self.dataset, desc=desc, unit="img"):
            self._save_one(img, target, name)

        self.lock_file.write_text(json.dumps({"version": self.VERSION, "date": create_time_id()}))

    def _save_one(self, img: Image, target: T, name: str):
        save_image(img, self.cache_dir / f"{name}-{str(target)}.jpg")
