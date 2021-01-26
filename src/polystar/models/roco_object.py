from dataclasses import dataclass
from enum import auto
from typing import Any, Dict, NewType

from polystar.models.box import Box
from polystar.utils.no_case_enum import NoCaseEnum

Json = NewType("Json", Dict[str, Any])

ArmorNumber = NewType("ArmorNumber", int)


class ArmorColor(NoCaseEnum):
    GREY = auto()
    BLUE = auto()
    RED = auto()

    UNKNOWN = auto()

    def __str__(self):
        return self.name.lower()

    @property
    def short(self) -> str:
        return self.name[0] if self != ArmorColor.UNKNOWN else "?"


class ArmorDigit(NoCaseEnum):  # CHANGING
    # Those have real numbers
    HERO = auto()
    # ENGINEER = 2
    STANDARD_1 = auto()
    STANDARD_2 = auto()
    # STANDARD_3 = 5

    # Those have symbols
    # OUTPOST = auto()
    # BASE = auto()
    # SENTRY = auto()

    UNKNOWN = auto()
    OUTDATED = auto()  # Old labelisation

    def __str__(self) -> str:
        # if self.value <= 5:
        #     return f"{self.value} ({self.name.title()})"
        # return self.name.title()
        return f"{self.digit} ({self.name.title()})"  # hacky, but a number is missing (2)

    @property
    def short(self) -> str:
        return self.digit if self != ArmorDigit.UNKNOWN else "?"

    @property
    def digit(self) -> int:
        return self.value + (self.value >= 2)  # hacky, but a number is missing (2)


class ObjectType(NoCaseEnum):
    CAR = auto()
    WATCHER = auto()
    BASE = auto()
    ARMOR = auto()
    IGNORE = auto()


@dataclass
class ROCOObject:
    type: ObjectType
    box: Box

    def __str__(self) -> str:
        return self.type.name


@dataclass
class Armor(ROCOObject):
    number: ArmorNumber
    type: ArmorDigit
    color: ArmorColor

    def __repr__(self):
        return f"<{self} {self.color} {self.number}>"


class ObjectFactory:
    @staticmethod
    def from_json(json: Json) -> ROCOObject:
        t: ObjectType = ObjectType(json["name"])

        x, y, w, h = (
            int(float(json["bndbox"]["xmin"])),
            int(float(json["bndbox"]["ymin"])),
            int(float(json["bndbox"]["xmax"])) - int(float(json["bndbox"]["xmin"])),
            int(float(json["bndbox"]["ymax"])) - int(float(json["bndbox"]["ymin"])),
        )

        x, y = max(0, x), max(0, y)

        if t is not ObjectType.ARMOR:
            return ROCOObject(type=t, box=Box.from_size(x, y, w, h=h))

        armor_number = ArmorNumber(int(json["armor_class"])) if json["armor_class"] != "none" else 0

        return Armor(
            type=t, box=Box.from_size(x, y, w, h=h), number=armor_number, color=ArmorColor(json["armor_color"])
        )

    @staticmethod
    def to_json(obj: ROCOObject) -> Json:
        rv = Json(
            {
                "name": obj.type.name.lower(),
                "bndbox": {"xmin": obj.box.x1, "xmax": obj.box.x2, "ymin": obj.box.y1, "ymax": obj.box.y2},
            }
        )
        if isinstance(obj, Armor):
            rv.update({"armor_class": obj.number, "armor_color": obj.color.name.lower()})
        return rv
