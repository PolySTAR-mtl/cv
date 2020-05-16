from abc import ABC
from typing import Any, Dict

from dataclasses import dataclass


class TargetABC(ABC):
    def to_json(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class SimpleTarget(TargetABC):
    d: float
    phi: float
    theta: float
