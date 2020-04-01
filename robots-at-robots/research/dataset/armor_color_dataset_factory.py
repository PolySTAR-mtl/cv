from pathlib import Path

from polystar.common.models.object import ArmorColor
from research.dataset.armor_image_dataset_factory import ArmorImageDatasetGenerator


class ArmorColorDatasetGenerator(ArmorImageDatasetGenerator[str]):
    task_name: str = "colors"

    def _label_from_armor_info(self, color: ArmorColor, digit: int, k: int, path: Path) -> str:
        return color.name
