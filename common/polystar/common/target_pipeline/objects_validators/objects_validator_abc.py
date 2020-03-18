from abc import abstractmethod, ABC
from typing import List

import numpy as np

from polystar.common.models.object import Object


class ObjectsValidatorABC(ABC):
    @abstractmethod
    def validate(self, objects: List[Object], image: np.ndarray) -> List[bool]:
        pass

    def filter(self, objects: List[Object], image: np.ndarray) -> List[Object]:
        return [obj for obj, is_valid in zip(objects, self.validate(objects, image)) if is_valid]
