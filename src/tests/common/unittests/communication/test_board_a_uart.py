from logging import ERROR
from unittest import TestCase

from serial import serial_for_url

from polystar.communication.board_a import BoardA, logger
from polystar.communication.command import Command, make_target_command
from polystar.target_pipeline.target_abc import SimpleTarget


class TestBoardA(TestCase):
    def setUp(self) -> None:
        self.serial = serial_for_url("loop://", timeout=0)
        self.board_a = BoardA(self.serial)

    def test_send_command(self):
        command = make_target_command(SimpleTarget(theta=0.25, phi=-0.35, d=5.3))

        self.board_a.send_command(command)

        self.assertEqual(b"\xfc\x02\x00\x07Y\xfa\x00\xa2\xfe\xb4\x14", self.serial.readall())

    def test_read_command(self):
        self.serial.write(b"\xfc\x01\x00\x01N")

        [command] = self.board_a.read_commands()

        self.assertEqual(Command(id=1, data=b"N"), command)

    def test_noise_before_command(self):
        with self.assertLogs(logger, ERROR):
            self.serial.write(b"NOISE\xfc\x01\x00\x01N")

            [command] = self.board_a.read_commands()

            self.assertEqual(Command(id=1, data=b"N"), command)

    def test_missing_data(self):
        with self.assertLogs(logger, ERROR):
            self.serial.write(b"\xfc\x01\x00\x02A")

            [] = self.board_a.read_commands()

            self.assertEqual(0, self.serial.in_waiting)
