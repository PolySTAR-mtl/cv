import json
from pathlib import Path
from shutil import rmtree
from typing import ClassVar, Generic, Optional

from polystar.common.models.image import Image, save_image
from polystar.common.utils.misc import identity
from polystar.common.utils.time import create_time_id
from polystar.common.utils.tqdm import smart_tqdm
from research.common.datasets.lazy_dataset import LazyDataset, TargetT
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.transform_dataset import TransformDataset
from research.robots_at_robots.dataset.armor_dataset_factory import ArmorDataset
from research.robots_at_robots.dataset.armor_value_target_factory import ArmorValueTargetFactory


class ArmorValueDatasetCache(Generic[TargetT]):
    VERSION: ClassVar[str] = "2.0"

    def __init__(
        self,
        roco_dataset_builder: ROCODatasetBuilder,
        cache_dir: Path,
        dataset_name: str,
        target_factory: ArmorValueTargetFactory[TargetT],
    ):
        self.target_factory = target_factory
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.roco_dataset_builder = roco_dataset_builder
        self.lock_file = cache_dir / ".lock"

    def generate_if_needed(self):
        cause = self._get_generation_cause()
        if cause is None:
            return
        self._clean_cache_dir()
        self.save(self._generate(), cause)

    def _clean_cache_dir(self):
        rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir()

    def save(self, dataset: LazyDataset[Image, TargetT], cause: str):
        desc = f"Generating dataset {self.dataset_name} (cause: {cause})"
        for img, target, name in smart_tqdm(dataset, desc=desc, unit="img"):
            save_image(img, self.cache_dir / f"{name}-{target}.jpg")
        self.lock_file.write_text(json.dumps({"version": self.VERSION, "date": create_time_id()}))

    def _generate(self) -> LazyDataset[Image, TargetT]:
        return TransformDataset(
            ArmorDataset(self.roco_dataset_builder.to_images().build_lazy()), identity, self.target_factory.from_armor
        )

    def _get_generation_cause(self) -> Optional[str]:
        if not self.lock_file.exists():
            return "lock not found"
        version = json.loads(self.lock_file.read_text())["version"]
        if version != self.VERSION:
            return f"upgrade [{version} -> {self.VERSION}]"
