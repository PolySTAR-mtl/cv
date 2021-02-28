from abc import ABC, abstractmethod
from typing import Iterable

from polystar.communication.command import NO_TARGET_COMMAND, Command, make_target_command
from polystar.target_pipeline.target_abc import SimpleTarget


class CSLinkABC(ABC):
    def read_commands(self) -> Iterable[Command]:
        return []

    @abstractmethod
    def send_command(self, command: Command):
        pass

    def send_target(self, target: SimpleTarget):
        self.send_command(make_target_command(target))

    def send_no_target(self):
        self.send_command(NO_TARGET_COMMAND)
