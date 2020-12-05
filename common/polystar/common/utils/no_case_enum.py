from enum import IntEnum


class NoCaseEnum(IntEnum):
    @classmethod
    def _missing_(cls, key):
        return cls[key.capitalize()]
