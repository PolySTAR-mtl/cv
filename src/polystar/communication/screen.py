from polystar.communication.command import Command
from polystar.communication.cs_link_abc import CSLinkABC


class Screen(CSLinkABC):
    def send_command(self, command: Command):
        print(command)
