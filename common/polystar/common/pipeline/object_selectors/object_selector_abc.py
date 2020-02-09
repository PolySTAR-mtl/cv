from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.common.models.object import Object


class ObjectSelectorABC(ABC):
    @abstractmethod
    def select(self, objects: List[Object], image: np.array) -> Object:
        pass
