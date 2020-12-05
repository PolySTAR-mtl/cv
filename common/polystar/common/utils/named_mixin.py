from typing import Any


class NamedMixin:
    @property
    def name(self) -> str:
        return getattr(self, "_name", self.__class__.__name__)

    @name.setter
    def name(self, name: str):
        self._name = name

    def __str__(self):
        return self.name


def get_name(example: Any) -> str:
    return getattr(example, "name", example.__class__.__name__)
