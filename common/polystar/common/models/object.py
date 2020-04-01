from dataclasses import dataclass
from enum import auto
from typing import Any, Dict, NewType

from polystar.common.utils.no_case_enum import NoCaseEnum

Json = NewType("Json", Dict[str, Any])

ArmorNumber = NewType("ArmorNumber", int)


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
    numero: ArmorNumber = -1
    color: ArmorColor = ArmorColor.Unknown


class ObjectFactory:
    @staticmethod
    def from_json(json: Json) -> Object:
        t: ObjectType = ObjectType(json["name"])

        x, y, w, h = (
            int(float(json["bndbox"]["xmin"])),
            int(float(json["bndbox"]["ymin"])),
            int(float(json["bndbox"]["xmax"])) - int(float(json["bndbox"]["xmin"])),
            int(float(json["bndbox"]["ymax"])) - int(float(json["bndbox"]["ymin"])),
        )

        x, y = max(0, x), max(0, y)

        if t is not ObjectType.Armor:
            return Object(type=t, x=x, y=y, w=w, h=h)

        armor_number = int(json["armor_class"]) if json["armor_class"] != "none" else 0

        return Armor(type=t, x=x, y=y, w=w, h=h, numero=armor_number, color=ArmorColor(json["armor_color"]))

    @staticmethod
    def to_json(obj: Object) -> Json:
        rv = {
            "name": obj.type.value.lower(),
            "bndbox": {"xmin": obj.x, "xmax": obj.x + obj.w, "ymin": obj.y, "ymax": obj.y + obj.h},
        }
        if isinstance(obj, Armor):
            rv.update({"armor_class": obj.numero, "armor_color": obj.color.value.lower()})
        return rv
