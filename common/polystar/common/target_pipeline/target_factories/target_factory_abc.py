from abc import ABC, abstractmethod

import numpy as np

from polystar.common.models.object import Object
from polystar.common.target_pipeline.target_abc import TargetABC


class TargetFactoryABC(ABC):
    @abstractmethod
    def from_object(self, obj: Object, image: np.array) -> TargetABC:
        pass
