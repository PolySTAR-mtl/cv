from abc import ABC, abstractmethod
from typing import Iterable

from polystar.communication.command import Command


class CSLinkABC(ABC):
    def read_commands(self) -> Iterable[Command]:
        return []

    @abstractmethod
    def send_command(self, command: Command):
        pass
