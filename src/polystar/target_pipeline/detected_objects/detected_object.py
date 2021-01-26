from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from polystar.models.roco_object import ROCOObject


@dataclass
class DetectedROCOObject(ROCOObject):
    confidence: float

    previous_occurrences: Deque["DetectedROCOObject"] = field(init=False, default_factory=deque)
    step_of_detection: int = -1

    def __str__(self) -> str:
        return f"{self.type.name} ({self.confidence:.1%})"
