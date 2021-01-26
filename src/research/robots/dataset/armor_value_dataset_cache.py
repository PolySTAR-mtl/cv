import json
from pathlib import Path
from shutil import rmtree
from typing import ClassVar, Generic, Optional

from google.cloud.exceptions import Forbidden

from polystar.models.image import Image, save_image
from polystar.utils.misc import identity
from polystar.utils.time import create_time_id
from polystar.utils.tqdm import smart_tqdm
from research.common.datasets.lazy_dataset import LazyDataset, TargetT
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.transform_dataset import TransformDataset
from research.common.gcloud.gcloud_storage import GCStorages
from research.robots.dataset.armor_dataset_factory import ArmorDataset
from research.robots.dataset.armor_value_target_factory import ArmorValueTargetFactory


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

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_or_download_if_needed(self):
        cause = self._get_generation_cause()
        if cause is None:
            return
        self._clean_cache_dir()
        try:
            GCStorages.DEV.download_directory(self.cache_dir)
            cause = self._get_generation_cause()
            if cause is None:
                return
            self._clean_cache_dir()
        except FileNotFoundError:
            cause += " and not on gcloud"
        except Forbidden:
            pass
        self.save(self._generate(), cause)

    def _clean_cache_dir(self):
        rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir()

    def save(self, dataset: LazyDataset[Image, TargetT], cause: str):
        desc = f"Generating dataset {self.dataset_name} (cause: {cause})"
        for img, target, name in smart_tqdm(dataset, desc=desc, unit="img"):
            save_image(img, self.cache_dir / f"{name}-{str(target)}.jpg")
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
