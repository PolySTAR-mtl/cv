from dataclasses import dataclass
from enum import auto
from typing import Any, Dict, NewType

from polystar.common.models.box import Box
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
    box: Box
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
            return Object(type=t, box=Box.from_size(x, y, w, h=h))

        armor_number = ArmorNumber(json["armor_class"]) if json["armor_class"] != "none" else 0

        return Armor(
            type=t, box=Box.from_size(x, y, w, h=h), numero=armor_number, color=ArmorColor(json["armor_color"])
        )

    @staticmethod
    def to_json(obj: Object) -> Json:
        rv = Json(
            {
                "name": obj.type.value.lower(),
                "bndbox": {"xmin": obj.box.x1, "xmax": obj.box.x2, "ymin": obj.box.y1, "ymax": obj.box.y2},
            }
        )
        if isinstance(obj, Armor):
            rv.update({"armor_class": obj.numero, "armor_color": obj.color.value.lower()})
        return rv
