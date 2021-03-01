from polystar.communication.command import Command
from polystar.communication.cs_link_abc import CSLinkABC


class TogglableCSLink(CSLinkABC):
    def __init__(self, cs_link: CSLinkABC, is_on: bool):
        self.is_on = is_on
        self.cs_link = cs_link

    def send_command(self, command: Command):
        if not self.is_on:
            return
        self.cs_link.send_command(command)

    def toggle(self):
        self.is_on = not self.is_on
