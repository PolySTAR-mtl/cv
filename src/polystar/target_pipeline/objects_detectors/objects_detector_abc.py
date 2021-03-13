from abc import ABC, abstractmethod
from typing import List

import numpy as np

from polystar.target_pipeline.detected_objects.objects_params import ObjectParams


class ObjectsDetectorABC(ABC):
    @abstractmethod
    def detect(self, image: np.array) -> List[ObjectParams]:
        pass
