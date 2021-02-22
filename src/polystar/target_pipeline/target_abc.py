from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict

from polystar.constants import BYTE_ORDER


class TargetABC(ABC):
    def to_json(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class SimpleTarget(TargetABC):
    d: float
    phi: float
    theta: float

    def __bytes__(self) -> bytes:
        return (
            int(self.theta * 1_000).to_bytes(length=2, byteorder=BYTE_ORDER)
            + int(self.phi * 1_000).to_bytes(length=2, byteorder=BYTE_ORDER, signed=True)
            + int(self.d * 1_000).to_bytes(length=2, byteorder=BYTE_ORDER)
        )
