from abc import abstractmethod
from pathlib import Path
from typing import Generic

from polystar.common.models.object import Armor
from research.common.datasets.lazy_dataset import TargetT


class ArmorValueTargetFactory(Generic[TargetT]):
    def from_file(self, file: Path) -> TargetT:
        return self.from_str(file.stem.split("-")[-1])

    @abstractmethod
    def from_str(self, label: str) -> TargetT:
        pass

    @abstractmethod
    def from_armor(self, armor: Armor) -> TargetT:
        pass
