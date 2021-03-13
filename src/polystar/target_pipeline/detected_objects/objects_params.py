from dataclasses import dataclass


@dataclass
class ObjectParams:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float
    object_class_id: int
