from dataclasses import dataclass
from enum import auto
from typing import Any, Dict, NewType, Set

from polystar.models.box import Box
from polystar.utils.no_case_enum import NoCaseEnum

Json = NewType("Json", Dict[str, Any])


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


VALID_NUMBERS_2021: Set[int] = {1, 3, 4}  # University League # CHANGING


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

    @staticmethod
    def from_number(n: int) -> "ArmorDigit":
        if n == 0:
            return ArmorDigit.UNKNOWN

        if n not in VALID_NUMBERS_2021:
            return ArmorDigit.OUTDATED

        return ArmorDigit(n - (n >= 3))  # hacky, but digit 2 is absent


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
    number: int
    digit: ArmorDigit
    color: ArmorColor

    def __repr__(self):
        return f"<{self} {self.color} {self.number}>"


class ROCOObjectFactory:
    def __init__(self, image_w: int, image_h: int):
        self.image_w = image_w
        self.image_h = image_h

    def from_json(self, json: Json) -> ROCOObject:
        t: ObjectType = ObjectType(json["name"])

        x1, y1, x2, y2 = (
            int(float(json["bndbox"]["xmin"])),
            int(float(json["bndbox"]["ymin"])),
            int(float(json["bndbox"]["xmax"])),
            int(float(json["bndbox"]["ymax"])),
        )

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(x2, self.image_w - 1), min(y2, self.image_h - 1)
        box = Box.from_positions(x1, y1, x2, y2)

        if t is not ObjectType.ARMOR:
            return ROCOObject(type=t, box=box)

        armor_number = int(json["armor_class"]) if json["armor_class"] != "none" else 0

        return Armor(
            type=t,
            box=box,
            number=armor_number,
            digit=ArmorDigit.from_number(armor_number),
            color=ArmorColor(json["armor_color"]),
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
