from enum import Enum


class NoCaseEnum(Enum):
    @classmethod
    def _missing_(cls, key):
        return cls[key.capitalize()]
