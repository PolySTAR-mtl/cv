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

    def __bytes__(self) -> bytes:
        return (
            int(self.theta * 1_000).to_bytes(length=2, byteorder="little")
            + int(self.phi * 1_000).to_bytes(length=2, byteorder="little", signed=True)
            + int(self.d * 1_000).to_bytes(length=2, byteorder="little")
        )
