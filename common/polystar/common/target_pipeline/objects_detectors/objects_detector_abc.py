from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.common.models.object import Object


class ObjectsDetectorABC(ABC):
    @abstractmethod
    def detect(self, image: np.array) -> List[Object]:
        pass
