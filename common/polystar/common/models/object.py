from dataclasses import dataclass
from enum import auto

from polystar.common.utils.no_case_enum import NoCaseEnum


class ObjectType(NoCaseEnum):
    Car = auto()
    Watcher = auto()
    Base = auto()
    Armor = auto()
    Ignore = auto()


@dataclass
class Object:
    type: ObjectType

    x: int
    y: int
    w: int
    h: int

    confidence: float = 1


class ArmorColor(NoCaseEnum):
    Grey = auto()
    Blue = auto()
    Red = auto()
    Unknown = auto()


@dataclass
class Armor(Object):
    numero: int = -1
    color: ArmorColor = ArmorColor.Unknown
