from dataclasses import dataclass, field
from typing import List

from polystar.common.models.image import Image


@dataclass
class PointOfInterest:
    x: int
    y: int


@dataclass
class LabeledImage:
    image: Image
    point_of_interests: List[PointOfInterest] = field(default_factory=list)
