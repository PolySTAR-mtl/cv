from pathlib import Path

from polystar.common.models.object import ArmorColor
from research.dataset.armor_image_dataset_factory import ArmorImageDatasetGenerator


class ArmorColorDatasetGenerator(ArmorImageDatasetGenerator[str]):
    def _label(self, color: ArmorColor, digit: int, k: int, path: Path) -> str:
        return color.name
