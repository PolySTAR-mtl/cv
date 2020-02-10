from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict


class TargetABC(ABC):
    def to_json(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class SimpleTarget(TargetABC):
    d: float
    phi: float
    theta: float
