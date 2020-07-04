from pathlib import Path

from polystar.common.models.object import Armor
from research.robots_at_robots.dataset import ArmorImageDatasetGenerator


class ArmorColorDatasetGenerator(ArmorImageDatasetGenerator[str]):
    task_name: str = "colors"

    def _label_from_armor_info(self, armor: Armor, k: int, path: Path) -> str:
        return armor.color.name
