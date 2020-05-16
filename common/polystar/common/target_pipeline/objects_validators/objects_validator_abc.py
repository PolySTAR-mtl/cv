from abc import abstractmethod, ABC
from typing import List, TypeVar, Generic

import numpy as np

from polystar.common.models.object import Object

ObjectT = TypeVar("ObjectT", bound=Object)


class ObjectsValidatorABC(Generic[ObjectT], ABC):
    def filter(self, objects: List[ObjectT], image: np.ndarray) -> List[ObjectT]:
        return [obj for obj, is_valid in zip(objects, self.validate(objects, image)) if is_valid]

    def validate(self, objects: List[ObjectT], image: np.ndarray) -> List[bool]:
        return [self.validate_single(obj, image) for obj in objects]

    @abstractmethod
    def validate_single(self, obj: ObjectT, image: np.ndarray) -> bool:
        pass
