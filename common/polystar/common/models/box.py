from dataclasses import dataclass, field
from math import sqrt
from typing import List

from memoized_property import memoized_property

from polystar.common.models.object import Object


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    x1: int = field(repr=False)
    y1: int = field(repr=False)
    x2: int = field(repr=False)
    y2: int = field(repr=False)

    @memoized_property
    def area(self) -> int:
        return self.w * self.h

    def convex_hull_with(self, box: "Box") -> "Box":
        return Box.from_positions(
            min(self.x1, box.x1), min(self.y1, box.y1), max(self.x2, box.x2), max(self.y2, box.y2)
        )

    def area_intersection(self, box: "Box") -> int:
        x1 = max(box.x1, self.x1)
        y1 = max(box.y1, self.y1)
        x2 = min(box.x2, self.x2)
        y2 = min(box.y2, self.y2)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def distance_among_axis(self, box: "Box", axis: int) -> float:
        if axis == 0:
            p11, p12, p21, p22 = self.x1, self.x2, box.x1, box.x2
        elif axis == 1:
            p11, p12, p21, p22 = self.y1, self.y2, box.y1, box.y2
        (p11, p12), (p21, p22) = sorted(((p11, p12), (p21, p22)))
        return max(0, p21 - p12)

    @staticmethod
    def from_positions(x1: int, y1: int, x2: int, y2: int) -> "Box":
        return Box(x=x1, y=y1, h=y2 - y1, w=x2 - x1, x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def from_size(x: int, y: int, w: int, h: int) -> "Box":
        return Box(x=x, y=y, w=w, h=h, x1=x, y1=y, x2=x + w, y2=y + h)


def get_all_object_boxes(objects: List[Object]) -> List[Box]:
    return [Box.from_size(o.x, o.y, o.w, o.h) for o in objects]
