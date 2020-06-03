from collections import deque
from typing import Deque

from dataclasses import dataclass, field

from polystar.common.models.object import Object


@dataclass
class DetectedObject(Object):
    confidence: float

    previous_occurrences: Deque["DetectedObject"] = field(init=False, default_factory=deque)
    step_of_detection: int = -1

    def __str__(self) -> str:
        return f"{self.type.name} ({self.confidence:.1%})"
