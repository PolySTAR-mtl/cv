from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class Set(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name.lower()

    __str__ = __repr__
