import logging
from typing import ClassVar, Iterable

from serial import Serial

from polystar.communication.command import Command
from polystar.communication.cs_link_abc import CSLinkABC
from polystar.constants import BYTE_ORDER

logger = logging.getLogger(__name__)


class BoardA(CSLinkABC):  # TODO: multi-thread /// Or increase BR ?
    START_BYTE: ClassVar[bytes] = b"\xfc"

    def __init__(self, serial: Serial):
        self.serial = serial

    def __del__(self):
        self.serial.close()

    def send_command(self, command: Command):
        self.serial.write(
            self.START_BYTE
            + command.id.to_bytes(length=2, byteorder=BYTE_ORDER)
            + command.size.to_bytes(length=1, byteorder=BYTE_ORDER)
            + bytes(command.data)
        )

    def read_commands(self) -> Iterable[Command]:
        while self.serial.in_waiting:
            try:
                yield self._read_command()
            except (StartingByteError, NoEnoughDataError):
                logger.exception("Communication issue with Board A")

    def _read_command(self) -> Command:
        self._check_starting_byte()
        command_id = self._read_next_int(size=2)
        size = self._read_next_int(size=1)
        data = self._read_bytes(size)
        return Command(command_id, data)

    def _check_starting_byte(self):
        starting_byte = self._read_bytes(1)
        if starting_byte != self.START_BYTE:
            raise StartingByteError(starting_byte)

    def _read_next_int(self, size: int) -> int:
        return int.from_bytes(self._read_bytes(size), byteorder=BYTE_ORDER)

    def _read_bytes(self, size: int) -> bytes:
        if size > self.serial.in_waiting:
            self.serial.reset_input_buffer()
            raise NoEnoughDataError(self.serial.in_waiting, size)
        return self.serial.read(size=size)


class StartingByteError(ValueError):
    def __init__(self, starting_byte: bytes):
        super().__init__(f"got {starting_byte}")


class NoEnoughDataError(BufferError):
    def __init__(self, remaining: int, expected: int):
        super().__init__(f"Only {remaining} remaining bytes, while expecting {expected}")
