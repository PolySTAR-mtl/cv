from dataclasses import dataclass
from enum import Enum, auto


class ObjectType(Enum):
    Car = auto()
    Watcher = auto()
    Base = auto()
    Armor = auto()
    Sentry = auto()


@dataclass
class Object:
    type: ObjectType

    confidence: float

    x: int
    y: int
    w: int
    h: int
